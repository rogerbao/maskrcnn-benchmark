from maskrcnn_benchmark.config import cfg
from predictor import COCODemo
import cv2
import os
import torch
config_file = "../configs/e2e_mask_rcnn_R_50_FPN_1x.yaml"

# update the config options with the config file
cfg.merge_from_file(config_file)
# manual override some options
# cfg.merge_from_list(["MODEL.DEVICE", "cpu"])

torch.cuda.set_device(3)

coco_demo = COCODemo(
    cfg,
    min_image_size=800,
    confidence_threshold=0.7,
)
# load image and then run prediction

img_path = '../Dataset/HumanCollection_mini/coco/HC_mini_test/'
pred_path = 'result_pred/'
img_list = os.listdir(img_path)

for image_name in img_list:
    img = cv2.imread(img_path + image_name)
    predictions = coco_demo.run_on_opencv_image(img)
    cv2.imwrite(pred_path + image_name.replace('.jpg', '.png'), predictions)
