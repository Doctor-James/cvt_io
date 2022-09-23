from hydra import core, initialize, compose
from omegaconf import OmegaConf

import sys
sys.path.append("/home/jialvzou/cvt_io")


# CHANGE ME
DATASET_DIR = '/data2/zjl/nuScenes'
LABELS_DIR = '/data2/zjl/nuScenes/cvt_labels_nuscenes_v2'


core.global_hydra.GlobalHydra.instance().clear()        # required for Hydra in notebooks

initialize(config_path='../config')

# Add additional command line overrides
cfg = compose(
    config_name='config',
    overrides=[
        'experiment.save_dir=../logs/',                 # required for Hydra in notebooks
        '+experiment=cvt_nuscenes_vehicle',
        f'data.dataset_dir={DATASET_DIR}',
        f'data.labels_dir={LABELS_DIR}',
        'data.version=v1.0-trainval',
        'loader.batch_size=1',
    ]
)

# resolve config references
OmegaConf.resolve(cfg)

print(list(cfg.keys()))
print(cfg['visualization'])

import torch
import numpy as np

from cross_view_transformer.common import setup_experiment, load_backbone
from cross_view_transformer.utils.instance import colorize_displacement


# Additional splits can be added to cross_view_transformer/data/splits/nuscenes/
SPLIT = 'val'
SUBSAMPLE = 1
IF_QUICK_EVAL = True


model, data, viz = setup_experiment(cfg)

dataset = data.get_split(SPLIT, loader=False)
if IF_QUICK_EVAL:
    SPLIT = 'val_qualitative_000'
    SUBSAMPLE = 5
dataset = data.get_split(SPLIT, loader=False)
dataset = torch.utils.data.ConcatDataset(dataset)
dataset = torch.utils.data.Subset(dataset, range(0, len(dataset), SUBSAMPLE))

loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2)

print(len(dataset))

from pathlib import Path


# Download a pretrained model (13 Mb)
# MODEL_URL = 'https://www.cs.utexas.edu/~bzhou/cvt/cvt_nuscenes_vehicles_50k.ckpt'
# CHECKPOINT_PATH = '../logs/cvt_nuscenes_vehicles_50k.ckpt'
# CHECKPOINT_PATH = '/data/zhulianghui/ProjectB_BEV/cross_view_transformers/logs/cross_view_transformers_test/version_16/checkpoints/model-v1.ckpt'
# CHECKPOINT_PATH = '/data/zhulianghui/ProjectX_Transformer/cross_view_transformers+/logs/cross_view_transformers_test_ori/version_1/checkpoints/model-v1.ckpt'
# CHECKPOINT_PATH = '/data/zhulianghui/ProjectB_BEV/cross_view_transformers/logs/cross_view_transformers_test/version_20/checkpoints/model-v1.ckpt'
# CHECKPOINT_PATH = '/data/zhulianghui/ProjectB_BEV/cross_view_transformers/logs/cross_view_transformers_test/version_49_offset0.01/checkpoints/model-v1.ckpt'
# CHECKPOINT_PATH = '/data/zhulianghui/ProjectB_BEV/cross_view_transformers/logs/cross_view_transformers_test/version_56/checkpoints/model.ckpt'
# CHECKPOINT_PATH = '/data/zhulianghui/ProjectB_BEV/cross_view_transformers/logs/cross_view_transformers_test/version_56_offset0.01_true/checkpoints/model-v1.ckpt'
CHECKPOINT_PATH = '/home/jialvzou/cvt_io/logs/cross_view_transformers_test/version_1/checkpoints/model.ckpt'
# !wget $MODEL_URL -O $CHECKPOINT_PATH


if Path(CHECKPOINT_PATH).exists():
    network = load_backbone(CHECKPOINT_PATH)
else:
    network = model.backbone

    print(f'{CHECKPOINT_PATH} not found. Using randomly initialized weights.')


import torch
import time
import imageio
# import ipywidgets as widgets


# GIF_PATH = './predictions.gif'

print("decide device...")

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = 'cuda:0'
print(device)

network.to(device)
network.eval()

images = list()

print("loader len is : ", len(loader))

from cross_view_transformer.metrics import PanopticMetric
from cross_view_transformer.utils.instance import predict_instance_segmentation_and_trajectories
import time

from thop import profile

metric_panoptic_val = PanopticMetric(n_classes=2)

limit = 1

with torch.no_grad():
    for batch in loader:

        limit -= 1

        if limit < 0:
            break

        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        # # test cost time
        # torch.cuda.synchronize()
        # start = time.time()
        # for i in range(5000):
        #     pred = network(batch)
        #
        # torch.cuda.synchronize()
        # end = time.time()
        # total_time = end - start
        # print('total_time:{:.2f}'.format(total_time))

        # test flops and params
        flops, params = profile(network, inputs=(batch,))
        print(flops)
        print(params)


        #         print('pred keys: ', pred.keys())
        #         print('batch keys: ', batch.keys())

#         # seg
#         # pred['segmentation'] = torch.sigmoid(pred['bev'].unsqueeze(1).cpu()) # pred
#         pred['segmentation'] = batch['bev'][:, 4:8, :, :].sum(1, keepdim=False).unsqueeze(1).unsqueeze(1).cpu()  # gt
#
#         pred['segmentation'][pred['segmentation'] > 1] = 1
#         thresh = 0.5 * torch.ones_like(pred['segmentation'], device=pred['segmentation'].device)
#         pred['segmentation'] = torch.cat((thresh, pred['segmentation']), 2) # pred['segmentation'] shape: 1x1x1x200x200 => 1x1x2x1x200x200
#
#         # center
#         # pred['center'] = torch.sigmoid(pred['center']) # pred
#         pred['center'] = batch['center']  # gt
#         pred['center'] = batch['centerness_bev']  # gt
#         pred['instance_center'] = pred['center'].unsqueeze(1).cpu()
#
#         if False:
#             pos_center_below = torch.where(pred['instance_center'] < 0.5)
#             pred['instance_center'][pos_center_below] = 0
#
#         # offset
#         pred['instance_offset'] = batch['offset_bev'].unsqueeze(0).cpu()  # gt
#         if False:
#             # pred['instance_offset'] = pred['instance_offset'] * 0.1
#             pos0 = torch.where(pred['instance_offset'] == 0)
#             pred['instance_offset'][pos0] = 255.0
#         # pred['instance_offset'] = pred['offset'].unsqueeze(0).cpu() # pred
#
#         # no flow
#         pred['instance_flow'] = None
#
#         # inference ins
#         pred_consistent_instance_seg, centers = predict_instance_segmentation_and_trajectories(
#             pred, compute_matched_centers=False, make_consistent=True, if_return_centers=True
#         )
#
#         # visualize
#         if True:
#             import matplotlib.pyplot as plt
#             plt.imshow(pred['segmentation'][0, 0, 1, :, :].cpu().numpy())
#             plt.show()
#
#             plt.imshow(pred['instance_center'][0, 0, 0, :, :].cpu().numpy())
#             plt.show()
#
#             plt.imshow(pred_consistent_instance_seg[0, 0, :, :].cpu().numpy())
#             plt.show()
#
#             plt.imshow(batch['instance_bev'][0, 0, :, :].cpu().numpy())
#             plt.show()
#
#
#             pos = torch.where(pred['segmentation'][0, 0, 1, :, :] > 0.5)
#             offset_w_seg = pred['instance_offset'][0, 0, :, :]
#             offset_mask = torch.zeros_like(offset_w_seg)
#             offset_mask[:, pos[0], pos[1]] = 1
#
#             offset_w_seg = offset_w_seg * offset_mask
#
#             plt.imshow(colorize_displacement(offset_w_seg.cpu().numpy(), if_scale=True))
#             plt.show()
#
#         # ins gt process
#         ins_gt = batch['instance_bev']
#         ins_gt_unique = torch.unique(batch['instance_bev'])
#         ins_gt_int = torch.zeros_like(batch['instance_bev'], dtype=torch.int64)
#         for i in range(len(ins_gt_unique)):
#             pos = torch.where(ins_gt == ins_gt_unique[i])
#             ins_gt_int[pos] = i
#
#         # visibility process
#         mask = batch['visibility'] <= 1
#         mask = mask[:, None].expand_as(pred_consistent_instance_seg)  # b c h w
#
#         pred_consistent_instance_seg[mask] = 0 # m
#         ins_gt_int[mask] = 0
#
#         # center out of range process
#
#
#         print("mask sum: ", mask.sum())
#
#         # visualize
#         if False:
#             import matplotlib.pyplot as plt
#             # plt.imshow(pred_consistent_instance_seg[0, 0, :, :].cpu().numpy())
#             # plt.show()
#
#             plt.imshow(ins_gt_int[0, 0, :, :].cpu().numpy())
#             plt.show()
#
#             plt.imshow(mask[0, 0, :, :].cpu().numpy())
#             plt.show()
#
#         # compute ins metrics
#         metric_panoptic_val.update_(pred_consistent_instance_seg.cpu(), ins_gt_int.cpu())
#         # metric_panoptic_val.update_(pred.cpu(), label.cpu())
#         # metric_panoptic_val.update_(ins_gt_int.cpu(), ins_gt_int.cpu())
#
#         # metric_panoptic_val(pred, batch)
#
# scores = metric_panoptic_val.compute()
# print('metric fn: ', metric_panoptic_val.false_negative)
# print('metric fp: ', metric_panoptic_val.false_positive)
# print('metric tp: ', metric_panoptic_val.true_positive)
# print('iou :', metric_panoptic_val.iou)
#
# print("instance scores: ", scores)
#
# import sys;
#
# sys.exit()