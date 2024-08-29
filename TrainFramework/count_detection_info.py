import os
from FB_detector import FB_detector
from utils.common import load_data, GetMiddleImg_ModelInput_for_MatImageList
from config.opts import opts
import numpy as np


os.environ['KMP_DUPLICATE_LIB_OK']='True'
def IOU(box1, box2):
    """
        计算IOU
    """
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]

    inter_rect_x1 = max(b1_x1, b2_x1)
    inter_rect_y1 = max(b1_y1, b2_y1)
    inter_rect_x2 = min(b1_x2, b2_x2)
    inter_rect_y2 = min(b1_y2, b2_y2)

    inter_area = max(inter_rect_x2 - inter_rect_x1 + 1, 0) * \
                 max(inter_rect_y2 - inter_rect_y1 + 1, 0)
                 
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)

    return iou

num_to_english_c_dic = {3:"three", 5:"five", 7:"seven", 9:"nine", 11:"eleven"}
iou_threshold = 0.2

if __name__ == "__main__":
    opt = opts().parse()
    model_input_size = (int(opt.model_input_size.split("_")[0]), int(opt.model_input_size.split("_")[1])) # H,W

    input_img_num = opt.input_img_num
    aggregation_output_channels = opt.aggregation_output_channels
    aggregation_method = opt.aggregation_method
    input_mode=opt.input_mode
    backbone_name = opt.backbone_name
    fusion_method = opt.fusion_method
    # assign_method: The label assign method. binary_assign, guassian_assign or auto_assign
    if opt.assign_method == "binary_assign":
        abbr_assign_method = "ba"
    elif opt.assign_method == "guassian_assign":
        abbr_assign_method = "ga"
    elif opt.assign_method == "auto_assign":
        abbr_assign_method = "aa"
    else:
        raise("Error! abbr_assign_method error.")
    
    Add_name=opt.Add_name
    model_name=opt.model_name

    # FB_detector parameters
    # model_input_size=(384,672),
    # input_img_num=5, aggregation_output_channels=16, aggregation_method="multiinput", input_mode="GRG", backbone_name="cspdarknet53",
    # Add_name="as_1021_1", model_name="FB_object_detect_model.pth",
    # scale=80.
    
    fb_detector = FB_detector(model_input_size=model_input_size,
                              input_img_num=input_img_num, aggregation_output_channels=aggregation_output_channels,
                              aggregation_method=aggregation_method, input_mode=input_mode, backbone_name=backbone_name, fusion_method=fusion_method,
                              abbr_assign_method=abbr_assign_method, Add_name=Add_name, model_name=model_name, scale=opt.scale_factor)
    Cuda = True
    annotation_path = "./dataloader/" + "img_label_" + num_to_english_c_dic[input_img_num] + "_continuous_difficulty_val.txt"
    dataset_image_path = opt.data_root_path + "images/val/"

    True_detection = 0
    False_detection = 0
    total_bbox = 0


    with open(annotation_path) as f:
        lines = f.readlines()
    for line in lines:

        images, bboxes, _ = load_data(line, dataset_image_path, frame_num=input_img_num)
        total_bbox += len(bboxes)

        raw_image_shape = np.array(images[0].shape[0:2]) # h,w
        write_img, model_input= GetMiddleImg_ModelInput_for_MatImageList(images, model_input_size=model_input_size, continus_num=input_img_num, input_mode=input_mode)
        outputs = fb_detector.detect_image(model_input, raw_image_shape=raw_image_shape)
        detect_bboxes = outputs[0][:,:4]

        if len(bboxes) == 0:
            False_detection += len(detect_bboxes)
            continue
        log_bbox = [0] * len(bboxes)
        for detect_bbox in detect_bboxes:
            best_iou=0
            for idx,gt_bbox in enumerate(bboxes):
                iou=IOU(detect_bbox,gt_bbox)
                if iou >best_iou:
                    best_iou=iou
                    best_gt_idx=idx
            if best_iou>iou_threshold:
                if log_bbox[best_gt_idx] == 0:
                    True_detection += 1
                    log_bbox[best_gt_idx] = 1
                else:
                    False_detection += 1
            else:
                False_detection += 1
    print("total_bbox: ", total_bbox)
    print("True_detection: ", True_detection)
    print("False_background: ", total_bbox-True_detection)
    print("False_detection: ", False_detection)
            