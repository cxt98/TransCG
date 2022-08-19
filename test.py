"""
Testing scripts.

Authors: Hongjie Fang.
"""
import os
import yaml
import torch
import logging
import warnings
import argparse
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from utils.logger import ColoredLogger
from utils.builder import ConfigBuilder
from utils.functions import to_device
from time import perf_counter
import cv2


logging.setLoggerClass(ColoredLogger)
logger = logging.getLogger(__name__)
warnings.simplefilter("ignore", UserWarning)

parser = argparse.ArgumentParser()
parser.add_argument(
    '--cfg', '-c', 
    default = os.path.join('./configs', 'clearpose.yaml'), 
    help = 'path to the configuration file', 
    type = str
)
args = parser.parse_args();
cfg_filename = args.cfg

with open(cfg_filename, 'r') as cfg_file:
    cfg_params = yaml.load(cfg_file, Loader = yaml.FullLoader)

builder = ConfigBuilder(**cfg_params)

logger.info('Building models ...')

model = builder.get_model()

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model.to(device)

logger.info('Building dataloaders ...')
test_dataloader = builder.get_dataloader(split = 'test')

logger.info('Checking checkpoints ...')
stats_dir = builder.get_stats_dir()
# checkpoint_file = os.path.join(stats_dir, 'checkpoint-epoch3.tar')
checkpoint_file = cfg_params['checkpoint']['dir']
if os.path.isfile(checkpoint_file):
    checkpoint = torch.load(checkpoint_file)
    model.load_state_dict(checkpoint['model_state_dict'])
    start_epoch = checkpoint['epoch']
    logger.info("Checkpoint {} (epoch {}) loaded.".format(checkpoint_file, start_epoch))
else:
    raise FileNotFoundError('No checkpoint.')

metrics = builder.get_metrics()


def test():
    logger.info('Start testing process.')
    model.eval()
    metrics.clear()
    running_time = []
    with tqdm(test_dataloader) as pbar:
        for data_dict in pbar:
            data_path = data_dict['data_path']
            del data_dict['data_path']
            data_dict = to_device(data_dict, device)
            with torch.no_grad():
                time_start = perf_counter()
                res = model(data_dict['rgb'], data_dict['depth'])
                data_dict['pred'], data_dict['wmap'] = res
                for i in range(data_dict['pred'].shape[0]):
                    frame_id = int(data_path[i].split('/')[-1])
                    if frame_id > 1000: # currently do not need frame_id > 1000
                        continue
                    des = (data_dict['pred'][i].detach().cpu().numpy() * 1000).astype(np.uint16)
                    # dgt = (data_dict['depth_gt'][0].detach().cpu().numpy() * 1000).astype(np.uint16)
                    # mask = (data_dict['depth_gt_mask'][i].detach().cpu().numpy()*255).astype(np.uint8)
                    # raw = (data_dict['depth'][i].detach().cpu().numpy() * 1000).astype(np.uint16)
                    # dgt[mask != 255] = 0 
                    # des[mask != 255] = raw[mask != 255]
                    # print(np.abs(dgt.astype(np.float32)  - des.astype(np.float32)).sum()/np.sum(mask == 255))
                    # cv2.imwrite("gt.png", dgt)
                    # cv2.imwrite("{}-depth_pred.png".format(data_path[i]), des)
                    # cv2.imwrite("mask.png", mask)
                    # wmap = (data_dict['wmap'][i].detach().cpu().numpy() * 1000).astype(np.uint16)
                    # cv2.imwrite("{}-depth_wmap.png".format(data_path[i]), wmap)
                    cv2.imwrite("{}-depth_pred_mse.png".format(data_path[i]), des)
                
                time_end = perf_counter()
                # data_dict['pred'] = res
                _ = metrics.evaluate_batch(data_dict, record = True)
            duration = time_end - time_start
            pbar.set_description('Time: {:.4f}s'.format(duration))
            running_time.append(duration)
    avg_running_time = np.stack(running_time).mean()
    logger.info('Finish testing process, average running time: {:.4f}s'.format(avg_running_time))
    metrics_result = metrics.get_results()
    metrics.display_results()
    return metrics_result


if __name__ == '__main__':
    test()

