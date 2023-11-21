from FB_detector import FB_detector
from mAP import mean_average_precision
import os
import cv2
from PIL import Image
from queue import Queue
from utils.common import GetMiddleImg_ModelInput
import xml.etree.ElementTree as ET
from config.opts import opts
from utils.utils import FBObj
import numpy as np

classes=['bird']
def ConvertAnnotationLabelToFBObj(annotation_file, image_id):
    label_obj_list = []
    in_file = open(annotation_file, encoding='utf-8')
    tree=ET.parse(in_file)
    root = tree.getroot()

    for obj in root.iter('object'):
        cls = obj.find('name').text
        if cls not in classes:
            continue
        xmlbox = obj.find('bndbox')
        bbox = (int(xmlbox.find('xmin').text), int(xmlbox.find('ymin').text), int(xmlbox.find('xmax').text), int(xmlbox.find('ymax').text))
        label_obj_list.append(FBObj(score=1.0, image_id=image_id, bbox=bbox))
    return label_obj_list

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

    
    label_path = opt.data_root_path + "val/labels/" #.xlm label file path

    video_path = opt.data_root_path + "val/video/"
    video_name = opt.video_name

    continus_num = input_img_num


    image_q = Queue(maxsize=continus_num)
    cap=cv2.VideoCapture(video_path + video_name)
    frame_id = 0

    all_label_obj_list = []
    label_name_list=os.listdir(label_path)

    all_obj_result_list = []

    while (True):
        ret,frame=cap.read()
        if ret != True:
            break
        else:
            frame_id += 1
            image = Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
            image_q.put(image)
            image_shape = np.array(np.shape(image)[0:2]) # image size is 1280,720; image array's shape is 720,1280
            if frame_id >= continus_num:

                exist_label = False
                ### The output of detector is start from continus_num-int(continus_num/2) frame.
                # frame_id_str = "%06d" % int(frame_id-int(continus_num/2))
                frame_id_str = "%06d" % int((frame_id-1)-int(continus_num/2)) #The frame id in dataset start from 0, but this script start from 1.
                label_name = video_name.split(".")[0] + "_" + frame_id_str + ".xml"
                if label_name in label_name_list:
                    exist_label = True
                    all_label_obj_list += ConvertAnnotationLabelToFBObj(label_path + label_name, (frame_id-int(continus_num/2)))

                # If there's no label for the middle frame of this input quene, continue this detection.
                if exist_label == False:
                    _ = image_q.get()
                    continue
                _, model_input = GetMiddleImg_ModelInput(image_q, model_input_size=model_input_size, continus_num=continus_num, input_mode=input_mode)
                _ = image_q.get()
                outputs = fb_detector.detect_image(model_input, raw_image_shape=image_shape)

                if outputs != None:
                    obj_result_list = []
                    for output in outputs:
                        box = output[:4]
                        score = output[4]
                        ### The output of detector is start from continus_num-int(continus_num/2) frame.
                        obj_result_list.append(FBObj(score=score, image_id=(frame_id-int(continus_num/2)), bbox=box))
                    all_obj_result_list += obj_result_list
    AP_50,REC_50,PRE_50=mean_average_precision(all_obj_result_list,all_label_obj_list,iou_threshold=0.5)
    print("AP_50,REC_50,PRE_50:")
    print(AP_50,REC_50,PRE_50)
    AP_75,REC_75,PRE_75=mean_average_precision(all_obj_result_list,all_label_obj_list,iou_threshold=0.75)
    print("AP_75,REC_75,PRE_75:")
    print(AP_75,REC_75,PRE_75)
    mAP = 0
    for i in range(50,95,5):
        iou_t = i/100
        mAP_, _, _ = mean_average_precision(all_obj_result_list,all_label_obj_list,iou_threshold=iou_t)
        mAP += mAP_
    mAP = mAP/10
    print("mAP = ",mAP)