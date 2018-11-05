#!/bin/bash

python tools/train_net.py --config-file "configs/e2e_mask_rcnn_R_50_FPN_1x.yaml" --local_rank 3 \
 SOLVER.BASE_LR 0.0025 SOLVER.MAX_ITER 260000 SOLVER.STEPS "(160000, 200000)" MODEL.ROI_MASK_HEAD.RESOLUTION 224 MODEL.ROI_MASK_HEAD.LARGE_LOSS True\
 TASKINFO.TASKNAME "R50_FPN-large_loss_new"\

#python tools/train_net.py --config-file "configs/e2e_mask_rcnn_R_50_FPN_1x.yaml" --local_rank 2 \
# SOLVER.BASE_LR 0.0025 SOLVER.MAX_ITER 260000 SOLVER.STEPS "(160000, 200000)" MODEL.ROI_MASK_HEAD.RESOLUTION 28 MODEL.ROI_MASK_HEAD.LARGE_LOSS False\
# TASKINFO.TASKNAME "R50_FPN"\


#  MODEL.WEIGHT "snapshot/R50_FPN-large_loss/model_0150000.pth"\