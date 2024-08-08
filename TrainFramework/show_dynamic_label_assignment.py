import cv2
import os
import torch
from FB_detector import FB_detector
from utils.common import load_data_resize, GetMiddleImg_ModelInput_for_MatImageList
from config.opts import opts
import numpy as np
from utils.getDynamicTargets import getTargets
import copy


os.environ['KMP_DUPLICATE_LIB_OK']='True'


num_to_english_c_dic = {1:"one", 3:"three", 5:"five", 7:"seven", 9:"nine", 11:"eleven"}

def draw_heatmap(img, heatmap, image_size):
    heatmap_shape = heatmap.shape
    for i in range(heatmap_shape[0]): # height
        for j in range(heatmap_shape[1]): # weight
            if heatmap[i][j] > 0:
                guassion_value = heatmap[i][j]
                ref_point_position = []
                ref_point_position.append(j*(image_size[1]/heatmap_shape[1]) + (image_size[1]/heatmap_shape[1])/2) #### x
                ref_point_position.append(i*(image_size[0]/heatmap_shape[0]) + (image_size[0]/heatmap_shape[0])/2) #### y
                color_value = (255-int(255*guassion_value**2), 255-int(255*guassion_value**2), 255)
                # color_value = (int(255*guassion_value), int(255*guassion_value), int(255*guassion_value))
                cv2.circle(img, (int(ref_point_position[0]), int(ref_point_position[1])), 1, color_value, -1, cv2.LINE_AA)

if __name__ == "__main__":
    opt = opts().parse()
    model_input_size = (int(opt.model_input_size.split("_")[0]), int(opt.model_input_size.split("_")[1])) # H,W

    input_img_num = opt.input_img_num
    aggregation_output_channels = opt.aggregation_output_channels
    aggregation_method = opt.aggregation_method
    input_mode=opt.input_mode
    backbone_name = opt.backbone_name
    fusion_method = opt.fusion_method
    # assign_method: The label assign must be auto_assign, because we will show the dynamic label
    if opt.assign_method == "auto_assign":
        abbr_assign_method = "aa"
    else:
        raise("Error! assign_method isn't 'auto_assign'.")
    
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
    annotation_path = "./dataloader/" + "img_label_" + num_to_english_c_dic[input_img_num] + "_continuous_difficulty_train.txt"
    dataset_image_path = opt.data_root_path + "images/train/"

    
    with open(annotation_path) as f:
        lines = f.readlines()
    for line in lines:

        images, bboxes, first_img_name = load_data_resize(line, dataset_image_path, frame_num=input_img_num, model_input_size=(model_input_size[1], model_input_size[0]))
        middleimg, model_input= GetMiddleImg_ModelInput_for_MatImageList(images, model_input_size=model_input_size, continus_num=input_img_num, input_mode=input_mode)
        draw_bboxes = copy.deepcopy(bboxes)
        for box in draw_bboxes:
            cv2.rectangle(middleimg,(int(box[0]),int(box[1])),(int(box[2]),int(box[3])),(0,255,0),2)#x1,y1,x2,y2
            
        bs_bboxes = np.expand_dims(bboxes, axis=0)
        bs_bboxes = torch.from_numpy(bs_bboxes)
        get_target = getTargets(model_input_size=(model_input_size[1], model_input_size[0]), scale=opt.scale_factor) ###w,h

        predictions = fb_detector.inference(model_input)
        dynamic_labels = get_target(predictions, bs_bboxes)

        label_cls = dynamic_labels[0][0]  ### h,w,c
        label_conf = torch.sum(label_cls[:,:,1:], dim=2) # h, w ## Conf

        test_img = np.full((model_input_size[0],model_input_size[1],3), 255)
        test_img = test_img.astype(np.uint8)
        draw_heatmap(test_img, label_conf, image_size=model_input_size)
        for box in draw_bboxes:
            cv2.rectangle(test_img,(int(box[0]),int(box[1])),(int(box[2]),int(box[3])),(0,255,0),2)#x1,y1,x2,y2

        first_img_num_str = first_img_name.split(".")[0].split("_")[-1]
        num_str = "%06d" % int(input_img_num/2)
        img_name = first_img_name.split(first_img_num_str)[0] + num_str

        cv2.imwrite("./test_output/test_gt.png", middleimg)
        cv2.imwrite("./test_output/test_label_assign.png", test_img)
        str = input("Enter your input: ")
        if str == "q":
            is_quit = True
            break
        elif str == "s":
            cv2.imwrite("./test_output/" + img_name + "_gt.png", middleimg)
            cv2.imwrite("./test_output/" + img_name + "_label_assign.png", test_img)