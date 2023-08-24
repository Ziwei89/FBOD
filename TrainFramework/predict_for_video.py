import numpy as np
import cv2
import os
from FB_detector import FB_detector
from utils.common import GetMiddleImg_ModelInput
from config.opts import opts
from queue import Queue
from PIL import Image
import copy
import xml.etree.ElementTree as ET

os.environ['KMP_DUPLICATE_LIB_OK']='True'

classes=['bird']
def ConvertAnnotationLabelToBboxes(annotation_file):
    bboxes = []
    in_file = open(annotation_file, encoding='utf-8')
    tree=ET.parse(in_file)
    root = tree.getroot()

    for obj in root.iter('object'):
        cls = obj.find('name').text
        if cls not in classes:
            continue
        xmlbox = obj.find('bndbox')
        bbox = (int(xmlbox.find('xmin').text), int(xmlbox.find('ymin').text), int(xmlbox.find('xmax').text), int(xmlbox.find('ymax').text))
        bboxes.append(bbox)
    return bboxes


num_to_english_c_dic = {3:"three", 5:"five", 7:"seven", 9:"nine", 11:"eleven"}

if __name__ == "__main__":
    opt = opts().parse()

    raw_image_shape = (int(opt.raw_image_shape.split("_")[0]), int(opt.raw_image_shape.split("_")[1])) # H,W
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
    
    video_path = opt.video_path
    label_path = opt.data_label_path + "val/" #.xlm label file path
    label_name_list=os.listdir(label_path)

    video_name = opt.video_name
    
    # FB_detector parameters
    # raw_image_shape=(720,1280), model_input_size=(384,672),
    # input_img_num=5, aggregation_output_channels=16, aggregation_method="multiinput", input_mode="GRG", backbone_name="cspdarknet53",
    # Add_name="as_1021_1", model_name="FB_object_detect_model.pth",
    # scale=80.
    fb_detector = FB_detector(raw_image_shape=raw_image_shape, model_input_size=model_input_size,
                              input_img_num=input_img_num, aggregation_output_channels=aggregation_output_channels,
                              aggregation_method=aggregation_method, input_mode=input_mode, backbone_name=backbone_name, fusion_method=fusion_method,
                              abbr_assign_method=abbr_assign_method, Add_name=Add_name, model_name=model_name)
    
    Save_video_name = "./test_output/" + video_name.split(".")[0] + "_out.mp4"
    fps = 25

    video_dir = os.path.join("./", Save_video_name)
    videowriter = cv2.VideoWriter(video_dir, cv2.VideoWriter_fourcc(*"mp4v"), 25, (raw_image_shape[1], raw_image_shape[0])) ## need w,h

    image_q = Queue(maxsize=input_img_num)
    cap=cv2.VideoCapture(video_path + video_name)
    frame_id = 0

    while (True):
        ret,frame=cap.read()
        if ret != True:
            break
        else:
            frame_id += 1
            image = Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
            image_q.put(image)
            if frame_id >= input_img_num:

                ### The output of first stage is start from continus_num-int(continus_num/2) frame.
                # frame_id_str = "%05d" % int(frame_id-int(continus_num/2))
                frame_id_str = "%05d" % int((frame_id-1)-int(input_img_num/2)) #The frame id in dataset start from 0, but this script start from 1.
                label_name = video_name.split(".")[0] + "_" + frame_id_str + ".xml"
                label_bboxes = []
                if label_name in label_name_list:
                    label_bboxes = ConvertAnnotationLabelToBboxes(label_path + label_name)
                

                middle_img_, model_input = GetMiddleImg_ModelInput(image_q, model_input_size=model_input_size, continus_num=input_img_num, input_mode=input_mode)
                write_img = copy.copy(middle_img_)
                image_opencv = cv2.cvtColor(np.asarray(write_img),cv2.COLOR_RGB2BGR) 
                _ = image_q.get()
                outputs = fb_detector.detect_image(model_input)

                detect_bboxes = outputs[0][:,:4]
                for box in detect_bboxes:
                    cv2.rectangle(image_opencv,(int(box[0]),int(box[1])),(int(box[2]),int(box[3])),(0,0,255),2)#x1,y1,x2,y2
                for label_box in label_bboxes:
                    cv2.rectangle(image_opencv,(int(label_box[0]),int(label_box[1])),(int(label_box[2]),int(label_box[3])),(0,255,0),2)#x1,y1,x2,y2
                videowriter.write(image_opencv)
    videowriter.release()