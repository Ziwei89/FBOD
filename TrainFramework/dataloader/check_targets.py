# -*- coding: utf-8 -*-
import sys
sys.path.append("..")
import numpy as np
from torch.utils.data import DataLoader
import cv2
from dataloader.dataset_bbox import CustomDataset
import copy
import math


def is_point_in_bbox(point, bbox):
    condition1 = (point[0] >= bbox[0]-bbox[2]/2) and (point[0] <= bbox[0]+bbox[2]/2)
    condition2 = (point[1] >= bbox[1]-bbox[3]/2) and (point[1] <= bbox[1]+bbox[3]/2)
    if condition1 and condition2:
        return True
    else:
        return False

def min_max_ref_point_index(bbox, output_feature, model_input_size):
    min_x = bbox[0] - bbox[2]/2
    min_y = bbox[1] - bbox[3]/2

    max_x = bbox[0] + bbox[2]/2
    max_y = bbox[1] + bbox[3]/2
    min_wight_index = math.floor(max((min_x*output_feature[0])/model_input_size[0] - 1/2,0))
    min_height_index = math.floor(max((min_y*output_feature[1])/model_input_size[1] - 1/2,0))

    max_wight_index = math.ceil(min((max_x*output_feature[0])/model_input_size[0] - 1/2,output_feature[0]-1))
    max_height_index = math.ceil(min((max_y*output_feature[1])/model_input_size[1] - 1/2,output_feature[1]-1))

    return (min_wight_index, min_height_index, max_wight_index, max_height_index)

       
def draw_label_points(bboxes, write_img, model_input_size=(672,384), default_inner_proportion=0.7, stride=2): ###

    out_feature_size = [model_input_size[0]/stride, model_input_size[1]/stride] ## feature_w,feature_h
    ###  bbox[0] x1, bbox[1] y1, bbox[2] x2, bbox[3] y2, bbox[4] class_id, bbox[5] object score (difficult) ###
    if len(bboxes) == 0:
      return write_img
    # convert x1,y1,x2,y2 to cx,cy,o_w,o_h
    bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 0] ## o_w
    bboxes[:, 3] = bboxes[:, 3] - bboxes[:, 1] ## o_h
    bboxes[:, 0] = bboxes[:, 0] + bboxes[:, 2] / 2 # cx
    bboxes[:, 1] = bboxes[:, 1] + bboxes[:, 3] / 2 # cy

    sample_position_list = []

    for bbox in bboxes:
        obj_area = bbox[2] * bbox[3]
        if obj_area == 0:
            continue

        ###  bbox[0] cx, bbox[1] cy, bbox[2] o_w, bbox[3] o_h, bbox[4] class_id, bbox[5] difficult ###
        inner_bbox = copy.deepcopy(bbox[:4]) ###  inner_bbox[0] cx, inner_bbox[1] cy, inner_bbox[2] o_w, inner_bbox[3] o_h
        inner_bbox[2] = inner_bbox[2] * default_inner_proportion ### o_w
        inner_bbox[3] = inner_bbox[3] * default_inner_proportion ### o_h

        min_wight_index, min_height_index, max_wight_index, max_height_index = min_max_ref_point_index(bbox,out_feature_size,model_input_size)
        for i in range(min_height_index, max_height_index+1):
            for j in range(min_wight_index, max_wight_index+1):
                ref_point_position = []
                ref_point_position.append(j*(model_input_size[0]/out_feature_size[0]) + (model_input_size[0]/out_feature_size[0])/2) #### x
                ref_point_position.append(i*(model_input_size[1]/out_feature_size[1]) + (model_input_size[1]/out_feature_size[1])/2) #### y

                if is_point_in_bbox(ref_point_position, inner_bbox):# The point is in inner bbox.
                    if (i,j) in sample_position_list:
                        cv2.circle(write_img, (int(ref_point_position[0]),int(ref_point_position[1])), 1, (255, 255, 255), -1, cv2.LINE_AA)
                    else:
                        sample_position_list.append((i,j))
                        cv2.circle(write_img, (int(ref_point_position[0]),int(ref_point_position[1])), 1, (0, 0, 255), -1, cv2.LINE_AA)

                else:# The point is out inner box.
                    if is_point_in_bbox(ref_point_position, bbox):# The point is in object bounding box but out inner box.
                        cv2.circle(write_img, (int(ref_point_position[0]),int(ref_point_position[1])), 1, (0, 255, 0), -1, cv2.LINE_AA)
    return write_img

def datasetImgTocv2Mat_RGB(image_datas):
    images = []
    rgb_img = []
    i = 0
    for image_data in image_datas:
        image_data = np.expand_dims(image_data, axis=0)
        rgb_img.append(image_data)
        i += 1
        if i == 3:
            rgb_img = np.concatenate(rgb_img, axis = 0)
            rgb_img = np.transpose(rgb_img* 255.0, (1, 2, 0))# HWC->CHW
            rgb_img = np.array(rgb_img, dtype=np.uint8)
            rgb_img = cv2.cvtColor(np.asarray(rgb_img),cv2.COLOR_RGB2BGR)
            images.append(rgb_img)
            i = 0
            rgb_img = []
    return images

def datasetImgTocv2Mat_GRG(image_datas, continues_num):
    # The middle image is RGB, and the others are gray.
    images=[]
    rgb_img = []
    for i, image_data in enumerate(image_datas):
        if i >= int(continues_num/2) and i <= int(continues_num/2) + 2:
            # RGB image
            image_data = np.expand_dims(image_data, axis=0)
            rgb_img.append(image_data)
            if i == int(continues_num/2) + 2: # Has finished Collecting all the channels of the RGB image.
                rgb_img = np.concatenate(rgb_img, axis = 0)

                rgb_img = np.transpose(rgb_img* 255.0, (1, 2, 0))# HWC->CHW
                rgb_img = np.array(rgb_img, dtype=np.uint8)
                rgb_img = cv2.cvtColor(np.asarray(rgb_img),cv2.COLOR_RGB2BGR)
                images.append(rgb_img)
        else:
            # Gray image
            image_data = np.expand_dims(image_data, axis=0)
            image_data = np.transpose(image_data* 255.0, (1, 2, 0))# HWC->CHW
            image = np.array(image_data, dtype=np.uint8)
            images.append(image)
    return images

if __name__ == '__main__':
    image_size = (672,384)
    input_mode = "RGR"
    continues_num=5
    dataset_image_path = "../../dataset/val/images/"
    data = open("./img_label_five_continuous_difficulty_val.txt").readlines()
    #################LOC

    stride = 2
    in_w = int(image_size[0]/stride)
    in_h = int(image_size[1]/stride)

    batch_size = 1
    

    train_data = CustomDataset(data, image_size, image_path=dataset_image_path,input_mode=input_mode,continues_num=continues_num)
    dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=False)
    print(len(dataloader))
    for i, item in enumerate(dataloader):
        if i > len(dataloader):
            break
        image_datas, targets, names = item
        print(names)
        # print(image_datas.size())
        if input_mode == "RGB":
            images = datasetImgTocv2Mat_RGB(image_datas[0])
        else:
            images = datasetImgTocv2Mat_GRG(image_datas[0], continues_num)
        
        write_img = images[int(continues_num/2)]
        write_img2 = copy.deepcopy(write_img)
        bboxes = targets[0]
        for box in bboxes:
            cv2.rectangle(write_img,(int(box[0]),int(box[1])),(int(box[2]),int(box[3])),(255,0,0),2)#x1,y1,x2,y2
        cv2.imwrite("./test_output_img/test.png", write_img)
        write_img=draw_label_points(bboxes=bboxes, write_img=write_img2)
        cv2.imwrite("./test_output_img/test2.png", write_img2)

        str = input("Enter your input: ")
        if str == "\n":
            continue
        if str == "q":
            break