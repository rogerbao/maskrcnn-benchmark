from maskrcnn_benchmark.config import cfg
from demo.predictor import COCODemo
import cv2
import os
import torch

config_file = "configs/e2e_mask_rcnn_R_50_FPN_1x.yaml"

# update the config options with the config file
cfg.merge_from_file(config_file)
# manual override some options
# cfg.merge_from_list(["MODEL.DEVICE", "cpu"])

task_name = "R50_FPN-large_loss_new"
iteration = 259999
cfg.MODEL.WEIGHT="snapshot/"+task_name+"/model_{}_{:07d}.pth".format(task_name, iteration)

torch.cuda.set_device(1)

coco_demo = COCODemo(
    cfg,
    min_image_size=800,
    confidence_threshold=0.7,
)
# load image and then run prediction

# image = 'image/3BR_IMG_20180320_135032.jpg'
# img = cv2.imread(image)
# predictions = coco_demo.run_on_opencv_image(img)
# cv2.imwrite('predction.png', predictions)


# img_path = '../Dataset/HumanCollection_mini/coco/HC_mini_test/'
img_path = '../Dataset/HW_test_set/coco/HW_test_set/'
pred_path = 'result_pred/HW_test_set_mini/'
method = task_name+'-'+str(iteration)
if os.path.exists(pred_path)==0:
    os.makedirs(pred_path)
img_list = os.listdir(img_path)

for image_name in img_list:
    print(image_name)
    img = cv2.imread(img_path + image_name)
    # img = cv2.resize(img,None,fx=0.1, fy = 0.1)
    # predictions = coco_demo.run_on_opencv_image(img)
    predictions = coco_demo.brd_run_on_opencv_image(img)
    cv2.imwrite(pred_path + method + image_name, predictions)
