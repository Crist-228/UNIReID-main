from prettytable import PrettyTable
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
import torch.nn.parallel
import numpy as np
import time
import os.path as op

# import clip
from datasets import build_dataloader
from processor.processor import do_inference
from utils.checkpoint import Checkpointer
from utils.logger import setup_logger
from model import build_model
from utils.metrics import Evaluator
import argparse
from utils.iotools import load_train_configs


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="TranTextReID Text")
    parser.add_argument("--config_file", default='/data/wenjunying/UNIReID/multimodality-RSTPReid/RSTPReid/20240308_221517_Ls_addfusion_itc/configs.yaml')
    args = parser.parse_args()
    args = load_train_configs(args.config_file)
 

    args.training = False
    logger = setup_logger('CLIP2ReID', save_dir=args.output_dir, if_train=args.training)
    logger.info(args)
    device = "cuda"
    
    test_img_loader, test_txt_loader, test_sketch_loader = build_dataloader(args)
    model = build_model(args)
    checkpointer = Checkpointer(model)
    checkpointer.load(f=op.join(args.output_dir, 'fusion_best.pth'))
    model.to(device)
    
    do_inference(args, model, test_img_loader, test_txt_loader, test_sketch_loader)