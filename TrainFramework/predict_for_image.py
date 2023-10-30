import cv2
import os
from FB_detector import FB_detector
from utils.common import load_data, GetMiddleImg_ModelInput_for_MatImageList
from config.opts import opts
import numpy as np


os.environ['KMP_DUPLICATE_LIB_OK']='True'


num_to_english_c_dic = {1:"one", 3:"three", 5:"five", 7:"seven", 9:"nine", 11:"eleven"}

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
                              abbr_assign_method=abbr_assign_method, Add_name=Add_name, model_name=model_name)
    Cuda = True
    annotation_path = "./dataloader/" + "img_label_" + num_to_english_c_dic[input_img_num] + "_continuous_difficulty_val.txt"
    dataset_image_path = opt.data_path + "val/images/"

    
    with open(annotation_path) as f:
        lines = f.readlines()
    for line in lines:

        images, bboxes, _ = load_data(line, dataset_image_path, frame_num=input_img_num)
        raw_image_shape = np.array(images[0].shape[0:2]) # h,w
        # print("raw_image_shape:")
        # print(raw_image_shape)
        write_img, model_input= GetMiddleImg_ModelInput_for_MatImageList(images, model_input_size=model_input_size, continus_num=input_img_num, input_mode=input_mode)
        outputs = fb_detector.detect_image(model_input, raw_image_shape=raw_image_shape)

        detect_bboxes = outputs[0][:,:4]
        print("detect_bboxes")
        print(detect_bboxes)
        print("bboxes")
        print(bboxes)
        for box in detect_bboxes:
            cv2.rectangle(write_img,(int(box[0]),int(box[1])),(int(box[2]),int(box[3])),(0,0,255),2)#x1,y1,x2,y2
        for box in bboxes:
            cv2.rectangle(write_img,(int(box[0]),int(box[1])),(int(box[2]),int(box[3])),(0,255,0),2)#x1,y1,x2,y2
        cv2.imwrite("./test_output/test.png", write_img)

        str = input("Enter your input: ")
        if str == "q":
            is_quit = True
            break
        
