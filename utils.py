import logging
from torch.nn.utils import clip_grad_norm_

import torch
import torch.nn as nn

import os


def create_logger(log_dir='./logs', log_file='train.log'):
    if log_dir is None:
        log_dir = './logs'

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_file_path = os.path.join(log_dir, log_file)

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    fh = logging.FileHandler(log_file_path)
    fh.setLevel(logging.DEBUG)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger


def update_ema(model, ema_model, decay):
    with torch.no_grad():
        ema_params = ema_model.state_dict()
        model_params = model.state_dict()
        for key in model_params.keys():
            ema_params[key].data.mul_(decay).add_(model_params[key].data, alpha=1 - decay)

def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag

import torch.distributed as dist

def cleanup():
    dist.destroy_process_group()

from torch.utils.tensorboard import SummaryWriter

def create_tensorboard(log_dir):
    return SummaryWriter(log_dir)

def write_tensorboard(writer, log_data, step):
    for key, value in log_data.items():
        writer.add_scalar(key, value, step)
