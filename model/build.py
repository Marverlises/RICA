from model import objectives
from .clip_model import Transformer, QuickGELU, LayerNorm, build_CLIP_from_openai_pretrained, convert_weights, \
    Aggregator
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from collections import OrderedDict
from .losses import AlignmentContrastiveLoss, ContrastiveLoss, l2norm


class RICA(nn.Module):
    """RICA: Re-Ranking with Intra-Modal and Cross-Modal Alignment for Text-Based Person Search."""

    def __init__(self, args, num_classes=11003):
        super().__init__()
        self.args = args
        self.num_classes = num_classes
        self._set_task()
        
        CLIP_args = {'delta': args.delta, 'measure': args.measure, 'max_violation': args.max_violation,
                     'alignment_mode': args.alignment_mode, 'cap_dim': args.cap_dim, 'dropout': args.dropout,
                     'aggregation_type': args.aggregation_type, 'img_dim': args.img_dim, 'loss_names': args.loss_names}

        self.base_model, base_cfg = build_CLIP_from_openai_pretrained(args.pretrain_choice, args.img_size,
                                                                      args.stride_size, CLIP_alignment_args=CLIP_args)
        self.embed_dim = base_cfg['embed_dim']
        self.logit_scale = torch.ones([]) * (1 / args.temperature)

        if 'id' in args.loss_names:
            if 'gl' in self.current_task:
                self.alignment_criterion = AlignmentContrastiveLoss(margin=args.delta,
                                                                    measure=args.measure,
                                                                    max_violation=args.max_violation,
                                                                    aggregation=args.alignment_mode)
                self.matching_criterion = ContrastiveLoss(margin=args.delta,
                                                          measure=args.measure,
                                                          max_violation=args.max_violation,
                                                          sigma=args.sigma)

        if 'mlm' in args.loss_names:
            self.cross_attn = nn.MultiheadAttention(self.embed_dim,
                                                    self.embed_dim // 64,
                                                    batch_first=True)
            self.cross_modal_transformer = Transformer(width=self.embed_dim,
                                                       layers=args.cmt_depth,
                                                       heads=self.embed_dim //
                                                             64)
            scale = self.cross_modal_transformer.width ** -0.5

            self.ln_pre_t = LayerNorm(self.embed_dim)
            self.ln_pre_i = LayerNorm(self.embed_dim)
            self.ln_post = LayerNorm(self.embed_dim)
            
            proj_std = scale * ((2 * self.cross_modal_transformer.layers) ** -0.5)
            attn_std = scale
            fc_std = (2 * self.cross_modal_transformer.width) ** -0.5
            for block in self.cross_modal_transformer.resblocks:
                nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
                nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
                nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
                nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

            nn.init.normal_(self.cross_attn.in_proj_weight, std=attn_std)
            nn.init.normal_(self.cross_attn.out_proj.weight, std=proj_std)
            
            self.mlm_head = nn.Sequential(
                OrderedDict([('dense', nn.Linear(self.embed_dim, self.embed_dim)),
                             ('gelu', QuickGELU()),
                             ('ln', LayerNorm(self.embed_dim)),
                             ('fc', nn.Linear(self.embed_dim, args.vocab_size))]))
            
            nn.init.normal_(self.mlm_head.dense.weight, std=fc_std)
            nn.init.normal_(self.mlm_head.fc.weight, std=proj_std)


    def _set_task(self):
        """Set training tasks based on loss names."""
        loss_names = self.args.loss_names
        self.current_task = [l.strip() for l in loss_names.split('+')]
        print(f'Training Model with {self.current_task} tasks')


    def cross_former(self, q, k, v):
        x = self.cross_attn(
            self.ln_pre_t(q),
            self.ln_pre_i(k),
            self.ln_pre_i(v),
            need_weights=False)[0]
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.cross_modal_transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x)
        return x


    def encode_image(self, image):
        x, img_lengths = self.base_model.encode_image(image)
        if 'gl' in self.current_task:
            return x.float(), img_lengths
        else:
            return x[:, 0, :].float(), img_lengths


    def encode_text(self, text):
        x, text_lengths = self.base_model.encode_text(text)
        if 'gl' in self.current_task:
            return x.float(), text_lengths
        else:
            return x[torch.arange(x.shape[0]), text.argmax(dim=-1), :].float(), text_lengths

    def encode_text_single(self, text):
        x, text_lengths = self.base_model.encode_text(text)
        return x.float(), text_lengths


    def forward(self, batch):
        ret = dict()

        images = batch['images']
        caption_ids = batch['caption_ids']
        image_feats, text_feats, text_lengths, img_lengths = self.base_model(images, caption_ids)
        if 'gl' in self.current_task:
            i_feats = image_feats[:, 0, :].float()
            t_feats = text_feats[torch.arange(text_feats.shape[0]), caption_ids.argmax(dim=-1), :].float()
        else:
            i_feats = image_feats[:, 0, :].float()
            t_feats = text_feats[torch.arange(text_feats.shape[0]), caption_ids.argmax(dim=-1)].float()
        logit_scale = self.logit_scale
        ret.update({'temperature': 1 / logit_scale})

        if 'itc' in self.current_task:
            ret.update({'itc_loss': objectives.compute_itc(i_feats, t_feats, logit_scale)})

        if 'sdm' in self.current_task:
            ret.update({'sdm_loss': objectives.compute_sdm(i_feats, t_feats, batch['pids'], logit_scale)})

        if 'cmpm' in self.current_task:
            ret.update({'cmpm_loss': objectives.compute_cmpm(i_feats, t_feats, batch['pids'])})

        if 'id' in self.current_task:
            if 'gl' in self.current_task:
                new_global_img = l2norm(i_feats).float()
                new_image_feats = F.normalize(image_feats, p=2, dim=2).float()
                new_image_feats[:, 0, :] = new_global_img
                new_global_text = l2norm(t_feats).float()
                new_text_feats = F.normalize(text_feats, p=2, dim=2).float()
                new_text_feats[torch.arange(text_feats.shape[0]), caption_ids.argmax(dim=-1), :] = new_global_text

                # Handle duplicate person IDs
                pid = batch['pids']
                unique, counts = pid.unique(return_counts=True)
                duplicates = unique[counts > 1]
                duplicate_indices = {item: (pid == item).nonzero(as_tuple=True)[0].tolist() for item in duplicates}
                mask = torch.ones(new_global_img.size(0), dtype=torch.bool)
                for key, indices in duplicate_indices.items():
                    duplicate_rows_i = new_global_img[indices]
                    duplicate_rows_t = new_global_text[indices]
                    duplicate_rows_image = new_image_feats[indices]
                    duplicate_rows_text = new_text_feats[indices]
                    mean_vector_i = torch.mean(duplicate_rows_i, dim=0)
                    mean_vector_t = torch.mean(duplicate_rows_t, dim=0)
                    mean_vector_image = torch.mean(duplicate_rows_image, dim=0)
                    mean_vector_text = torch.mean(duplicate_rows_text, dim=0)
                    replacement_index = indices[0]
                    new_global_img[replacement_index] = mean_vector_i
                    new_global_text[replacement_index] = mean_vector_t
                    mask[indices[1:]] = False

                new_global_img = new_global_img[mask]
                new_global_text = new_global_text[mask]
                new_image_feats = new_image_feats[mask]
                new_text_feats = new_text_feats[mask]
                img_lengths = img_lengths[mask]
                text_lengths = text_lengths[mask]
                
                matching_loss = self.matching_criterion(new_global_img, new_global_text)
                ret.update({'matching_loss': matching_loss.float()})

                alignment_loss = self.alignment_criterion(new_image_feats, new_text_feats, img_lengths, text_lengths)
                ret.update({'alignment_loss': alignment_loss.float()})
            else:
                image_logits = self.classifier(i_feats.half()).float()
                text_logits = self.classifier(t_feats.half()).float()
                ret.update({'id_loss': objectives.compute_id(image_logits, text_logits,
                                                             batch['pids']) * self.args.id_loss_weight})

                image_pred = torch.argmax(image_logits, dim=1)
                text_pred = torch.argmax(text_logits, dim=1)
                image_precision = (image_pred == batch['pids']).float().mean()
                text_precision = (text_pred == batch['pids']).float().mean()
                ret.update({'img_acc': image_precision})
                ret.update({'txt_acc': text_precision})
        if 'mlm' in self.current_task:
            mlm_ids = batch['mlm_ids']

            mlm_feats, _ = self.base_model.encode_text(mlm_ids)
            x = self.cross_former(mlm_feats, image_feats, image_feats)
            x = self.mlm_head(x)
            scores = x.float().reshape(-1, self.args.vocab_size)
            mlm_labels = batch['mlm_labels'].reshape(-1)
            ret.update({'mlm_loss': objectives.compute_mlm(scores, mlm_labels) * self.args.mlm_loss_weight})
            pred = scores.max(1)[1]
            mlm_label_idx = torch.nonzero(mlm_labels)
            acc = (pred[mlm_label_idx] == mlm_labels[mlm_label_idx]).float().mean()
            ret.update({'mlm_acc': acc})

        return ret


def build_model(args, num_classes=11003):
    model = RICA(args, num_classes)
    convert_weights(model)
    return model
