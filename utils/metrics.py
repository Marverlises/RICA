from prettytable import PrettyTable
import torch
import numpy as np
import os
import torch.nn.functional as F
import logging

from model.losses import AlignmentContrastiveLoss


def alignment_rank(target_caption_indices, text_feats, img_feats,
                   text_lens, img_lens, most_relevant, sim_fun=None):
    '''
    This function aims to get the most relevant local features based on the global features
    '''
    reserved_index = target_caption_indices[most_relevant:]
    # get the most relevant global features top 50 based on indices
    target_caption_indices = target_caption_indices[:most_relevant]
    # get the most relevant local features based on the target_caption_indices
    target_img_feats = img_feats[target_caption_indices]
    target_img_lens = [img_lens[i] for i in target_caption_indices]
    text_feats = text_feats.unsqueeze(0)
    # get the similarity matrix
    text_lens = [text_lens]
    sim_matrix = sim_fun(target_img_feats, text_feats, target_img_lens, text_lens)
    # get the most relevant local features based on the similarity matrix
    indices = torch.argsort(sim_matrix.view(sim_matrix.shape[0]), dim=0, descending=True).tolist()
    new_top_index = torch.cat((target_caption_indices[indices], reserved_index))
    return new_top_index


def rank(similarity, q_pids, g_pids, max_rank=10, get_mAP=True, alignment=False, global_index=50, sim_function=None,
         text_feats=None, img_feats=None, text_lens=None, img_lens=None):
    # use global feature and local features
    if alignment:
        if get_mAP:
            indices = torch.argsort(similarity, dim=1, descending=True)
        else:
            # accelerate sort with topk
            _, indices = torch.topk(
                similarity, k=max_rank, dim=1, largest=True, sorted=True
            )  # q * topk
        # get the most relevant global features top 50 based on indices and then get the local features the result is (q, 50)
        for i in range(indices.shape[0]):
            indices[i] = alignment_rank(indices[i], text_feats[i, :, :], img_feats, text_lens[i], img_lens,
                                        most_relevant=50,
                                        sim_fun=sim_function)
        pred_labels = g_pids[indices.cpu()]  # q * k
        matches = pred_labels.eq(q_pids.view(-1, 1))  # q * k
        # cumsum is used to calculate the number of correct predictions
        all_cmc = matches[:, :max_rank].cumsum(1)  # cumulative sum
        all_cmc[all_cmc > 1] = 1
        all_cmc = all_cmc.float().mean(0) * 100
        # all_cmc = all_cmc[topk - 1]

        if not get_mAP:
            return all_cmc, indices

        num_rel = matches.sum(1)  # q
        tmp_cmc = matches.cumsum(1)  # q * k

        inp = [tmp_cmc[i][match_row.nonzero()[-1]] / (match_row.nonzero()[-1] + 1.) for i, match_row in
               enumerate(matches)]
        mINP = torch.cat(inp).mean() * 100

        tmp_cmc = [tmp_cmc[:, i] / (i + 1.0) for i in range(tmp_cmc.shape[1])]
        tmp_cmc = torch.stack(tmp_cmc, 1) * matches
        AP = tmp_cmc.sum(1) / num_rel  # q
        mAP = AP.mean() * 100

        return all_cmc, mAP, mINP, indices

    # the original version only use global feature
    else:
        if get_mAP:
            indices = torch.argsort(similarity, dim=1, descending=True)
        else:
            # acclerate sort with topk
            _, indices = torch.topk(
                similarity, k=max_rank, dim=1, largest=True, sorted=True
            )  # q * topk
        pred_labels = g_pids[indices.cpu()]  # q * k
        matches = pred_labels.eq(q_pids.view(-1, 1))  # q * k
        # cumsum is used to calculate the number of correct predictions
        all_cmc = matches[:, :max_rank].cumsum(1)  # cumulative sum
        all_cmc[all_cmc > 1] = 1
        all_cmc = all_cmc.float().mean(0) * 100
        # all_cmc = all_cmc[topk - 1]

        if not get_mAP:
            return all_cmc, indices

        num_rel = matches.sum(1)  # q
        tmp_cmc = matches.cumsum(1)  # q * k

        inp = [tmp_cmc[i][match_row.nonzero()[-1]] / (match_row.nonzero()[-1] + 1.) for i, match_row in
               enumerate(matches)]
        mINP = torch.cat(inp).mean() * 100

        tmp_cmc = [tmp_cmc[:, i] / (i + 1.0) for i in range(tmp_cmc.shape[1])]
        tmp_cmc = torch.stack(tmp_cmc, 1) * matches
        AP = tmp_cmc.sum(1) / num_rel  # q
        mAP = AP.mean() * 100

        return all_cmc, mAP, mINP, indices


class Evaluator():
    def __init__(self, img_loader, txt_loader, loss_names=None):
        self.img_loader = img_loader  # gallery
        self.txt_loader = txt_loader  # query
        self.logger = logging.getLogger("RICA.eval")
        self.loss_type = loss_names

    def _compute_embedding(self, model):
        model = model.eval()
        device = next(model.parameters()).device

        qids, gids, qfeats, gfeats, text_lens, img_lens = [], [], [], [], [], []
        # text
        for pid, caption in self.txt_loader:
            caption = caption.to(device)
            with torch.no_grad():
                text_feat, t_lens = model.encode_text(caption)
            qids.append(pid.view(-1))  # flatten
            qfeats.append(text_feat)
            text_lens.extend(t_lens.tolist())
        qids = torch.cat(qids, 0)
        qfeats = torch.cat(qfeats, 0)

        # image
        for pid, img in self.img_loader:
            img = img.to(device)
            with torch.no_grad():
                img_feat, i_lens = model.encode_image(img)
            gids.append(pid.view(-1))  # flatten
            gfeats.append(img_feat)
            img_lens.extend(i_lens.tolist())
        gids = torch.cat(gids, 0)
        gfeats = torch.cat(gfeats, 0)

        return qfeats, gfeats, qids, gids, text_lens, img_lens

    def eval(self, model, i2t_metric=False):
        # qfeats is the caption features[6156, 512], gfeats is the image features[3074, 512]
        qfeats, gfeats, qids, gids, text_lens, img_lens = self._compute_embedding(model)
        # p=2 indicates the L2 norm
        if 'gl' in self.loss_type:
            # change text_lens(list) to tensor
            text_lens = torch.tensor(text_lens).to(qfeats.device)
            text_global_feat = qfeats[torch.arange(qfeats.shape[0]), text_lens - 1, :].float()
            img_global_feat = gfeats[:, 0, :].float()
            # =============== two stage inference ==============
            # get global image and caption similarity
            similarity = text_global_feat @ img_global_feat.t()
            sim_matrix_fn = AlignmentContrastiveLoss(aggregation='MrSw',
                                                     return_similarity_mat=True)
            t2i_cmc, t2i_mAP, t2i_mINP, _ = rank(similarity=similarity, q_pids=qids, g_pids=gids, max_rank=10,
                                                 get_mAP=True, alignment=True, global_index=50,
                                                 sim_function=sim_matrix_fn, text_lens=text_lens, img_lens=img_lens,
                                                 text_feats=qfeats, img_feats=gfeats)
            t2i_cmc, t2i_mAP, t2i_mINP = t2i_cmc.numpy(), t2i_mAP.numpy(), t2i_mINP.numpy()
            table = PrettyTable(["task", "R1", "R5", "R10", "mAP", "mINP"])
            table.add_row(['t2i', t2i_cmc[0], t2i_cmc[4], t2i_cmc[9], t2i_mAP, t2i_mINP])

            if i2t_metric:
                i2t_cmc, i2t_mAP, i2t_mINP, _ = rank(similarity=similarity.t(), q_pids=gids, g_pids=qids, max_rank=10,
                                                     get_mAP=True, alignment=True, global_index=50,
                                                     sim_function=sim_matrix_fn, text_lens=text_lens, img_lens=img_lens,
                                                     text_feats=qfeats, img_feats=gfeats)
                i2t_cmc, i2t_mAP, i2t_mINP = i2t_cmc.numpy(), i2t_mAP.numpy(), i2t_mINP.numpy()
                table.add_row(['i2t', i2t_cmc[0], i2t_cmc[4], i2t_cmc[9], i2t_mAP, i2t_mINP])
            # table.float_format = '.4'
            table.custom_format["R1"] = lambda f, v: f"{v:.3f}"
            table.custom_format["R5"] = lambda f, v: f"{v:.3f}"
            table.custom_format["R10"] = lambda f, v: f"{v:.3f}"
            table.custom_format["mAP"] = lambda f, v: f"{v:.3f}"
            table.custom_format["mINP"] = lambda f, v: f"{v:.3f}"
            self.logger.info('\n' + str(table))
            return t2i_cmc[0]

        else:
            text_lens = torch.tensor(text_lens).to(qfeats.device)
            qfeats = qfeats[torch.arange(qfeats.shape[0]), text_lens - 1, :].float()
            gfeats = gfeats[:, 0, :].float()
            qfeats = F.normalize(qfeats, p=2, dim=1)  # text features
            gfeats = F.normalize(gfeats, p=2, dim=1)  # image features
            # get image and caption similarity
            similarity = qfeats @ gfeats.t()

            t2i_cmc, t2i_mAP, t2i_mINP, _ = rank(similarity=similarity, q_pids=qids, g_pids=gids, max_rank=10,
                                                 get_mAP=True)
            t2i_cmc, t2i_mAP, t2i_mINP = t2i_cmc.numpy(), t2i_mAP.numpy(), t2i_mINP.numpy()
            table = PrettyTable(["task", "R1", "R5", "R10", "mAP", "mINP"])
            table.add_row(['t2i', t2i_cmc[0], t2i_cmc[4], t2i_cmc[9], t2i_mAP, t2i_mINP])

            if i2t_metric:
                i2t_cmc, i2t_mAP, i2t_mINP, _ = rank(similarity=similarity.t(), q_pids=gids, g_pids=qids, max_rank=10,
                                                     get_mAP=True)
                i2t_cmc, i2t_mAP, i2t_mINP = i2t_cmc.numpy(), i2t_mAP.numpy(), i2t_mINP.numpy()
                table.add_row(['i2t', i2t_cmc[0], i2t_cmc[4], i2t_cmc[9], i2t_mAP, i2t_mINP])
            # table.float_format = '.4'
            table.custom_format["R1"] = lambda f, v: f"{v:.3f}"
            table.custom_format["R5"] = lambda f, v: f"{v:.3f}"
            table.custom_format["R10"] = lambda f, v: f"{v:.3f}"
            table.custom_format["mAP"] = lambda f, v: f"{v:.3f}"
            table.custom_format["mINP"] = lambda f, v: f"{v:.3f}"
            self.logger.info('\n' + str(table))

            return t2i_cmc[0]
