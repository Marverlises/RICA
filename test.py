from prettytable import PrettyTable
import os
import torch
import numpy as np
import time
import os.path as op

from datasets import build_dataloader
from processor.processor import do_inference
from utils.checkpoint import Checkpointer
from utils.logger import setup_logger
from model import build_model
from utils.metrics import Evaluator
import argparse
from utils.iotools import load_train_configs

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="RICA Test")
    parser.add_argument("--config_file", default='', help='Path to config file')
    parser.add_argument("--output_dir", default='', help='Path to model directory')
    args = parser.parse_args()
    
    if not args.config_file or not args.output_dir:
        print("Please provide --config_file and --output_dir arguments")
        exit(1)
        
    args = load_train_configs(args.config_file)
    args.training = False
    logger = setup_logger('RICA', save_dir=args.output_dir, if_train=args.training)
    logger.info(args)
    device = "cuda"

    # Handle parameter name changes
    if hasattr(args, 'margin'):
        args.delta = args.margin
        delattr(args, 'margin')
    if not hasattr(args, 'sigma'):
        args.sigma = 0.08

    test_img_loader, test_txt_loader, num_classes = build_dataloader(args)
    model = build_model(args, num_classes=num_classes)
    checkpointer = Checkpointer(model)
    checkpointer.load(f=op.join(args.output_dir, 'best.pth'))
    model.to(device)
    do_inference(model, test_img_loader, test_txt_loader)
