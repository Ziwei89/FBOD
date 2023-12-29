# -*- coding: utf-8 -*-
import os
import sys
sys.path.append("..")
import numpy as np
from torch.utils.data import Dataset
import cv2
import dataloader.augmentations as DataAug
from torch.utils.data import DataLoader
from PIL import Image
import math
import copy


def object_score_to_HElevel(object_score):
    if object_score>0.75: # easy
        return 0
    elif object_score>0.5 and object_score<=0.75: # geneal
        return 1
    elif object_score>0.25 and object_score<=0.5: # difficulty
        return 2
    elif object_score<=0.25: # very difficulty
        return 3
    else:
        raise print("object_score error: object_score=", object_score)


def is_point_in_ellipse(point, ellipse_parameters, guassion_variance):
    point = [point[0]-ellipse_parameters[0],point[1]-ellipse_parameters[1]]
    if ((point[0]**2)/(ellipse_parameters[2]**2) + (point[1]**2)/(ellipse_parameters[3]**2)) < 1:
        guassion_value = math.exp((-1)*(point[0]**2/(2*guassion_variance[0]**2)+ point[1]**2/(2*guassion_variance[1]**2)))
        return True, guassion_value
    else:
        return False, 0

def is_point_in_bbox(point, bbox):
    condition1 = (point[0] >= bbox[0]-bbox[2]/2) and (point[0] <= bbox[0]+bbox[2]/2)
    condition2 = (point[1] >= bbox[1]-bbox[3]/2) and (point[1] <= bbox[1]+bbox[3]/2)
    if condition1 and condition2:
        return True
    else:
        return False

def min_max_ref_point_index(bbox, output_feature, image_size):
    min_x = bbox[0] - bbox[2]/2
    min_y = bbox[1] - bbox[3]/2

    max_x = bbox[0] + bbox[2]/2
    max_y = bbox[1] + bbox[3]/2
    min_wight_index = math.floor(max((min_x*output_feature[0])/image_size[0] - 1/2,0))
    min_height_index = math.floor(max((min_y*output_feature[1])/image_size[1] - 1/2,0))

    max_wight_index = math.ceil(min((max_x*output_feature[0])/image_size[0] - 1/2,output_feature[0]-1))
    max_height_index = math.ceil(min((max_y*output_feature[1])/image_size[1] - 1/2,output_feature[1]-1))

    return (min_wight_index, min_height_index, max_wight_index, max_height_index)

class CustomDataset(Dataset):
    def __init__(self, train_lines, image_size, image_path, input_mode="GRG", continues_num=5, classes_num=2,
                       default_inner_proportion=0.7, default_guassion_value=0.3, stride=2, assign_method="guassian_assign",
                       data_augmentation=False):
        # input_mode: "RGB" or "GRG". "RGB" means all the image is rgb mode. "GRG" means that the middle image remains RGB,
        # and the others will be coverted to gray.
        self.train_lines = train_lines
        self.train_batches = len(train_lines)
        self.image_size = image_size#(672,384)#w,h
        self.image_path = image_path
        self.input_mode = input_mode
        self.frame_num = continues_num

        self.classes_num = classes_num # include the background
        self.default_inner_proportion = default_inner_proportion
        self.default_guassion_value = default_guassion_value

        self.out_feature_size = [self.image_size[0]/stride, self.image_size[1]/stride]  ## w,h
        self.size_per_ref_point = self.image_size[0]/self.out_feature_size[0]
        self.assign_method = assign_method

        self.data_augmentation=data_augmentation
        
    def __load_data(self, line):
        """line of train_lines was saved as 'image name, label'"""
        line =  line.split()
        first_img_name = line[0]
        first_img_num_str = first_img_name.split(".")[0].split("_")[-1]
        first_img_num = int(first_img_num_str)
        images = []
        for num in range(first_img_num, first_img_num + self.frame_num):
            num_str = "%06d" % int(num)
            img_name = first_img_name.split(first_img_num_str)[0] + num_str + ".jpg"
            image_full_name = os.path.join(self.image_path,img_name)
            image = cv2.imread(image_full_name)
            images.append(image)
        if  line[1:][0] == "None":
            bboxes = np.array([])
        else:
            bboxes = np.array([np.array(list(map(float, box.split(',')))) for box in line[1:]])
        if self.data_augmentation == True:
            images, bboxes = DataAug.RandomVerticalFilp()(np.copy(images), np.copy(bboxes))
            images, bboxes = DataAug.RandomHorizontalFilp()(np.copy(images), np.copy(bboxes))
            images, bboxes = DataAug.RandomCenterFilp()(np.copy(images), np.copy(bboxes))
            images, bboxes = DataAug.Noise()(np.copy(images), np.copy(bboxes))
            images, bboxes = DataAug.HSV()(np.copy(images), np.copy(bboxes))
            images, bboxes = DataAug.RandomCrop()(np.copy(images), np.copy(bboxes))
        images, bboxes = DataAug.Resize((self.image_size[1], self.image_size[0]), True)(np.copy(images), np.copy(bboxes))
        # print(bboxes)
        return images, bboxes

    def __getitem__(self, index):
        lines = self.train_lines
        img_list, y = self.__load_data(lines[index])
        if self.input_mode == "RGB":
            img_inp = self.__Cv2ToImage_Concate(img_list)
        else:
            img_inp = self.__Cv2ToImage_OnlyMidRGB_Concate(img_list)
        bboxes = copy.deepcopy(y)
        if self.assign_method == "guassian_assign":
            targets = self.__create_one_head_label_guassian(y) #
        elif self.assign_method == "binary_assign":
            targets = self.__create_one_head_label(y) # one head
        else:
            raise("Error! assign method error!")
        return img_inp, targets, bboxes, lines[index].split(".")[0]

    def __len__(self):  
        return self.train_batches
    
    def __Cv2ToImage_Concate(self,img_list):
        # Covert and Concate the image list to an images array.
        img_inp = []
        for img in img_list:# Differen't from second stage.   
            img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            img = np.array(img, dtype=np.float32)# img shape is h,w,c 384,672,3
            img_inp.append(np.transpose(img / 255.0, (2, 0, 1)))
        img_inp = np.array(img_inp)
        img_inp = np.concatenate(img_inp, axis = 0)
        return img_inp
    
    def __Cv2ToImage_OnlyMidRGB_Concate(self, img_list):
        # Covert and Concate the image list to an images array.
        # The middle image remains RGB, and the others will be coverted to gray.
        img_inp=[]
        for i, img in enumerate(img_list):
            img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            if i == int(self.frame_num/2):
                img = np.array(img, dtype=np.float32)# img shape is h,w,c 384,672,3
            else:
                img = np.array(img.convert('L'), dtype=np.float32)
                img = img.reshape(self.image_size[1],self.image_size[0],1)
            # print(img.shape)
            img_inp.append(np.transpose(img / 255.0, (2, 0, 1)))
        img_inp = np.array(img_inp, dtype=object)
        img_inp = np.concatenate(img_inp, axis = 0)
        return img_inp
    
    # Guassian distribution
    def __create_one_head_label_guassian(self, bboxes):
        if len(bboxes) > 0:
            # convert x1,y1,x2,y2 to cx,cy,w,h
            bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 0] ## w
            bboxes[:, 3] = bboxes[:, 3] - bboxes[:, 1] ## h
            bboxes[:, 0] = bboxes[:, 0] + bboxes[:, 2] / 2 # cx
            bboxes[:, 1] = bboxes[:, 1] + bboxes[:, 3] / 2 # cy

        class_obj_list = []
        for bbox in bboxes:
            obj_area = bbox[2] * bbox[3]
            if obj_area == 0:
                continue
            length_x_semi_axis = bbox[2]/2
            length_y_semi_axis = bbox[3]/2

            length_inner_ellipse_x_semi_axis = length_x_semi_axis * self.default_inner_proportion
            length_inner_ellipse_y_semi_axis = length_y_semi_axis * self.default_inner_proportion

            ellipse_parameters = [bbox[0],bbox[1], length_inner_ellipse_x_semi_axis,length_inner_ellipse_y_semi_axis]

            guassion_variance_x = (length_inner_ellipse_x_semi_axis**2/(2*(-1)*math.log(self.default_guassion_value)))**0.5
            guassion_variance_y = (length_inner_ellipse_y_semi_axis**2/(2*(-1)*math.log(self.default_guassion_value)))**0.5

            ###  bbox[0] cx, bbox[1] cy, bbox[2] w, bbox[3] h, bbox[4] class_id, bbox[5] difficult ###
            bbox_ellipse_guassion_value = [bbox] + [ellipse_parameters] + [[guassion_variance_x,guassion_variance_y]]
            
            
            lamda =  self.size_per_ref_point / obj_area**0.5 #This parameter is related to the size of the target, and the smaller the target, the larger the parameter. 
            bbox_ellipse_guassion_value = bbox_ellipse_guassion_value + [lamda]
            class_obj_list.append(bbox_ellipse_guassion_value)
            
        label_list = [] # The list has two members, class_label_map and points_label_map
        
        position_guassian_value_dic ={}
        class_label_map = np.array(([1.] + [0.] * (self.classes_num - 1))*int(self.out_feature_size[0])*int(self.out_feature_size[1]))
        points_label_map = np.array([1.]*6*int(self.out_feature_size[0])*int(self.out_feature_size[1]))
        if len(class_obj_list) == 0:
            label_list.append(class_label_map)
            label_list.append(points_label_map)
        else:
            for obj in class_obj_list:
                min_wight_index, min_height_index, max_wight_index, max_height_index = min_max_ref_point_index(obj[0],self.out_feature_size,self.image_size)
                for i in range(min_height_index, max_height_index+1):
                    for j in range(min_wight_index, max_wight_index+1):
                        ref_point_position = []
                        ref_point_position.append(j*(self.image_size[0]/self.out_feature_size[0]) + (self.image_size[0]/self.out_feature_size[0])/2) #### x
                        ref_point_position.append(i*(self.image_size[1]/self.out_feature_size[1]) + (self.image_size[1]/self.out_feature_size[1])/2) #### y


                        result = is_point_in_ellipse(ref_point_position, obj[1], obj[2])
                        if result[0]:# The point is in inner ellipse.
                            ### If the position has two guassian values, restore the larger one.
                            if (i,j) in position_guassian_value_dic:
                                if position_guassian_value_dic[(i,j)] >= result[1]: ### Maybe error
                                    continue
                                else:
                                    position_guassian_value_dic[(i,j)] = result[1]
                            else:
                                position_guassian_value_dic[(i,j)] = result[1]
                            class_label_map[(i*int(self.out_feature_size[0]) + j) * self.classes_num + 0] = 0
                            class_label_map[(i*int(self.out_feature_size[0]) + j) * self.classes_num + int(obj[0][4]) + 1] = result[1]

                            points_label_map[(i*int(self.out_feature_size[0]) + j) * 6 + 0] = obj[0][0] # cx
                            points_label_map[(i*int(self.out_feature_size[0]) + j) * 6 + 1] = obj[0][1] # cy
                            points_label_map[(i*int(self.out_feature_size[0]) + j) * 6 + 2] = obj[0][2] # w
                            points_label_map[(i*int(self.out_feature_size[0]) + j) * 6 + 3] = obj[0][3] # h
                            

                            points_label_map[(i*int(self.out_feature_size[0]) + j) * 6 + 4] = obj[0][5] # difficult
                            points_label_map[(i*int(self.out_feature_size[0]) + j) * 6 + 5] = obj[3] # lamda parameter
                        else:# The point is out inner ellipse.
                            if is_point_in_bbox(ref_point_position, obj[0]):# The point is in object bounding box but out inner ellipse.
                                class_label_map[(i*int(self.out_feature_size[0]) + j) * self.classes_num + 0] = 0
            label_list.append(class_label_map)
            label_list.append(points_label_map)
        return label_list

    # No difficulty difference and no guassian heatmap
    def __create_one_head_label(self, bboxes):
        if len(bboxes) > 0:
            # convert x1,y1,x2,y2 to cx,cy,w,h
            bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 0] ## w
            bboxes[:, 3] = bboxes[:, 3] - bboxes[:, 1] ## h
            bboxes[:, 0] = bboxes[:, 0] + bboxes[:, 2] / 2 # cx
            bboxes[:, 1] = bboxes[:, 1] + bboxes[:, 3] / 2 # cy
        
            

        class_obj_list = []
        for bbox in bboxes:
            obj_area = bbox[2] * bbox[3]
            if obj_area == 0:
                continue

            ###  bbox[0] cx, bbox[1] cy, bbox[2] w, bbox[3] h, bbox[4] class_id, bbox[5] difficult ###
            inner_bbox = copy.deepcopy(bbox[:4]) ###  inner_bbox[0] cx, inner_bbox[1] cy, inner_bbox[2] w, inner_bbox[3] h
            inner_bbox[2] = inner_bbox[2] * self.default_inner_proportion ### w
            inner_bbox[3] = inner_bbox[3] * self.default_inner_proportion ### h

            lamda =  self.size_per_ref_point / obj_area**0.5 #This parameter is related to the size of the target, and the smaller the target, the larger the parameter. 
            bbox_inner_bbox_lamda = [bbox] + [inner_bbox] + [lamda]
            class_obj_list.append(bbox_inner_bbox_lamda)
            
        label_list = [] # The list has two members, class_label_map and points_label_map
        sample_position_list = []
        class_label_map = np.array(([1.] + [0.] * (self.classes_num - 1))*int(self.out_feature_size[0])*int(self.out_feature_size[1]))
        points_label_map = np.array([1.]*6*int(self.out_feature_size[0])*int(self.out_feature_size[1]))
        if len(class_obj_list) == 0:
            label_list.append(class_label_map)
            label_list.append(points_label_map)
        else:
            for obj in class_obj_list:
                min_wight_index, min_height_index, max_wight_index, max_height_index = min_max_ref_point_index(obj[0],self.out_feature_size,self.image_size)
                for i in range(min_height_index, max_height_index+1):
                    for j in range(min_wight_index, max_wight_index+1):
                        ref_point_position = []
                        ref_point_position.append(j*(self.image_size[0]/self.out_feature_size[0]) + (self.image_size[0]/self.out_feature_size[0])/2) #### x
                        ref_point_position.append(i*(self.image_size[1]/self.out_feature_size[1]) + (self.image_size[1]/self.out_feature_size[1])/2) #### y

                        if is_point_in_bbox(ref_point_position, obj[1]):# The point is in inner bbox.
                            if (i,j) in sample_position_list:
                                for class_id_index in range(self.classes_num): # Ignore this point
                                    class_label_map[(i*int(self.out_feature_size[0]) + j) * self.classes_num + class_id_index] = 0
                                    #continue before 2023/08/03
                                continue ### debug error. 2023/08/03
                            else:
                                sample_position_list.append((i,j))
                                class_label_map[(i*int(self.out_feature_size[0]) + j) * self.classes_num + 0] = 0
                                class_label_map[(i*int(self.out_feature_size[0]) + j) * self.classes_num + int(obj[0][4]) + 1] = 1

                                points_label_map[(i*int(self.out_feature_size[0]) + j) * 6 + 0] = obj[0][0] # cx
                                points_label_map[(i*int(self.out_feature_size[0]) + j) * 6 + 1] = obj[0][1] # cy
                                points_label_map[(i*int(self.out_feature_size[0]) + j) * 6 + 2] = obj[0][2] # w
                                points_label_map[(i*int(self.out_feature_size[0]) + j) * 6 + 3] = obj[0][3] # h
                                

                                points_label_map[(i*int(self.out_feature_size[0]) + j) * 6 + 4] = obj[0][5] # difficult
                                points_label_map[(i*int(self.out_feature_size[0]) + j) * 6 + 5] = obj[2] # lamda parameter

                        else:# The point is out inner box.
                            if is_point_in_bbox(ref_point_position, obj[0]):# The point is in object bounding box but out inner box.
                                class_label_map[(i*int(self.out_feature_size[0]) + j) * self.classes_num + 0] = 0
            label_list.append(class_label_map)
            label_list.append(points_label_map)
        return label_list

def Distance(pointa,pointb):
    return math.sqrt((pointa[0]-pointb[0])**2+(pointa[1]-pointb[1])**2)

class CustomDataset_multi_head(Dataset):
    def __init__(self, train_lines, image_size, image_path, input_mode="GRG", continues_num=5, classes_num=2,
                       default_inner_proportion=0.7, stride=[32,8,2], obj_maxsize_scale=[256.,80.,48.], data_augmentation=False): ##large medium small
        # input_mode: "RGB" or "GRG". "RGB" means all the image is rgb mode. "GRG" means that the middle image remains RGB,
        # and the others will be coverted to gray.
        self.train_lines = train_lines
        self.train_batches = len(train_lines)
        self.image_size = image_size#(672,384)#w,h
        self.image_path = image_path
        self.input_mode = input_mode
        self.frame_num = continues_num
        

        self.classes_num = classes_num # include the background
        self.default_inner_proportion = default_inner_proportion
        self.out_feature_size = [[self.image_size[0]/s, self.image_size[1]/s] for s in stride]  ## w,h ##large medium small
        self.size_per_ref_point = [self.image_size[0]/obj[0] for obj in self.out_feature_size] ##large medium small
        self.obj_maxsize_scale = obj_maxsize_scale

        self.data_augmentation=data_augmentation

    def __load_data(self, line):
        """line of train_lines was saved as 'image name, label'"""
        line =  line.split()
        first_img_name = line[0]
        img_ext = first_img_name.split(".")[1]
        first_img_num_str = first_img_name.split(".")[0].split("_")[-1]
        first_img_num = int(first_img_num_str)
        images = []
        prefix_img_name = ""
        split_count = len(first_img_name.split(first_img_num_str))
        for i in range(split_count-2):
            prefix_img_name += first_img_name.split(first_img_num_str)[i] + first_img_num_str
        
        for num in range(first_img_num, first_img_num + self.frame_num):
            num_str = "%06d" % int(num)

            img_name = prefix_img_name + first_img_name.split(first_img_num_str)[split_count-2] + num_str + "." + img_ext
            
            image_full_name = os.path.join(self.image_path,img_name)
            image = cv2.imread(image_full_name)
            images.append(image)
        if  line[1:][0] == "None":
            bboxes = np.array([])
        else:
            bboxes = np.array([np.array(list(map(float, box.split(',')))) for box in line[1:]])
        if self.data_augmentation == True:
            images, bboxes = DataAug.RandomVerticalFilp()(np.copy(images), np.copy(bboxes))
            images, bboxes = DataAug.RandomHorizontalFilp()(np.copy(images), np.copy(bboxes))
            images, bboxes = DataAug.RandomCenterFilp()(np.copy(images), np.copy(bboxes))
            images, bboxes = DataAug.Noise()(np.copy(images), np.copy(bboxes))
            images, bboxes = DataAug.HSV()(np.copy(images), np.copy(bboxes))
            images, bboxes = DataAug.RandomCrop()(np.copy(images), np.copy(bboxes))
        images, bboxes = DataAug.Resize((self.image_size[1], self.image_size[0]), True)(np.copy(images), np.copy(bboxes))
        # print(bboxes)
        return images, bboxes

    def __getitem__(self, index):
        lines = self.train_lines
        img_list, y = self.__load_data(lines[index])
        if self.input_mode == "RGB":
            img_inp = self.__Cv2ToImage_Concate(img_list)
        else:
            img_inp = self.__Cv2ToImage_OnlyMidRGB_Concate(img_list)
        bboxes = copy.deepcopy(y)
        targets = self.__create_multi_head_label(y) # multi head
        return img_inp, targets, bboxes, lines[index].split(".")[0]

    def __len__(self):  
        return self.train_batches
    
    def __Cv2ToImage_Concate(self,img_list):
        # Covert and Concate the image list to an images array.
        img_inp = []
        for img in img_list:# Differen't from second stage.   
            img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            img = np.array(img, dtype=np.float32)# img shape is h,w,c 384,672,3
            img_inp.append(np.transpose(img / 255.0, (2, 0, 1)))
        img_inp = np.array(img_inp)
        img_inp = np.concatenate(img_inp, axis = 0)
        return img_inp
    
    def __Cv2ToImage_OnlyMidRGB_Concate(self, img_list):
        # Covert and Concate the image list to an images array.
        # The middle image remains RGB, and the others will be coverted to gray.
        img_inp=[]
        for i, img in enumerate(img_list):
            img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            if i == int(self.frame_num/2):
                img = np.array(img, dtype=np.float32)# img shape is h,w,c 384,672,3
            else:
                img = np.array(img.convert('L'), dtype=np.float32)
                img = img.reshape(self.image_size[1],self.image_size[0],1)
            # print(img.shape)
            img_inp.append(np.transpose(img / 255.0, (2, 0, 1)))
        img_inp = np.array(img_inp, dtype=object)
        img_inp = np.concatenate(img_inp, axis = 0)
        return img_inp
    
    # No difficulty difference and no guassian heatmap
    def __create_multi_head_label(self, bboxes):
        if len(bboxes) > 0:
            # convert x1,y1,x2,y2 to cx,cy,w,h
            bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 0] ## w
            bboxes[:, 3] = bboxes[:, 3] - bboxes[:, 1] ## h
            bboxes[:, 0] = bboxes[:, 0] + bboxes[:, 2] / 2 # cx
            bboxes[:, 1] = bboxes[:, 1] + bboxes[:, 3] / 2 # cy
        
            

        lms_obj_list = [[],[],[]] # large, medium, small
        for bbox in bboxes:
            obj_area = bbox[2] * bbox[3]
            if obj_area == 0:
                continue
            obj_scale = bbox[2]**2 + bbox[3]**2

            ###  bbox[0] cx, bbox[1] cy, bbox[2] w, bbox[3] h, bbox[4] class_id, bbox[5] difficult ###
            inner_bbox = copy.deepcopy(bbox[:4]) ###  inner_bbox[0] cx, inner_bbox[1] cy, inner_bbox[2] w, inner_bbox[3] h
            inner_bbox[2] = inner_bbox[2] * self.default_inner_proportion ### w
            inner_bbox[3] = inner_bbox[3] * self.default_inner_proportion ### h
            
            # To determin which layer the obj belones to.
            # self.obj_maxsize_scale is a list. # large, medium, small  ### 256, 80, 48
            # self.size_per_ref_point is a list. # large, medium, small  ### 32, 8, 2
            if obj_scale < (self.obj_maxsize_scale[2])**2:
                lamda =  self.size_per_ref_point[2] / obj_area**0.5 #This parameter is related to the size of the target, and the smaller the target, the larger the parameter. 
                bbox_inner_bbox_lamda = [bbox] + [inner_bbox] + [lamda]
                lms_obj_list[2].append(bbox_inner_bbox_lamda)  #small object
            elif obj_scale >= (self.obj_maxsize_scale[1])**2:
                lamda =  self.size_per_ref_point[0] / obj_area**0.5 #This parameter is related to the size of the target, and the smaller the target, the larger the parameter. 
                bbox_inner_bbox_lamda = [bbox] + [inner_bbox] + [lamda]
                lms_obj_list[0].append(bbox_inner_bbox_lamda)  #large object
            else:
                lamda =  self.size_per_ref_point[1] / obj_area**0.5 #This parameter is related to the size of the target, and the smaller the target, the larger the parameter. 
                bbox_inner_bbox_lamda = [bbox] + [inner_bbox] + [lamda]
                lms_obj_list[1].append(bbox_inner_bbox_lamda)  #medium object

        label_list = [] # The list has 3 lists members (means large, medium, small), each list has two members, class_label_map and points_label_map
        # self.out_feature_size is a list. # To predict large, medium, small objects (image_size/32, image_size/8, image_size/2)
        for index, obj_list in enumerate(lms_obj_list):
            sample_position_list = []
            class_label_map = np.array(([1.] + [0.] * (self.classes_num - 1))*int(self.out_feature_size[index][0])*int(self.out_feature_size[index][1]))
            points_label_map = np.array([1.]*6*int(self.out_feature_size[index][0])*int(self.out_feature_size[index][1]))
            if len(obj_list) == 0:
                label_list.append([class_label_map, points_label_map])
            else:
                for obj in obj_list:
                    min_wight_index, min_height_index, max_wight_index, max_height_index = min_max_ref_point_index(obj[0],self.out_feature_size[index],self.image_size)
                    for i in range(min_height_index, max_height_index+1):
                        for j in range(min_wight_index, max_wight_index+1):
                            ref_point_position = []
                            ref_point_position.append(j*(self.image_size[0]/self.out_feature_size[index][0]) + (self.image_size[0]/self.out_feature_size[index][0])/2) #### x
                            ref_point_position.append(i*(self.image_size[1]/self.out_feature_size[index][1]) + (self.image_size[1]/self.out_feature_size[index][1])/2) #### y

                            if is_point_in_bbox(ref_point_position, obj[1]):# The point is in inner bbox.
                                if (i,j) in sample_position_list:
                                    for class_id_index in range(self.classes_num): # Ignore this point
                                        class_label_map[(i*int(self.out_feature_size[index][0]) + j) * self.classes_num + class_id_index] = 0
                                        #continue before 2023/08/03
                                    continue ### debug error. 2023/08/03
                                else:
                                    sample_position_list.append((i,j))
                                    class_label_map[(i*int(self.out_feature_size[index][0]) + j) * self.classes_num + 0] = 0
                                    class_label_map[(i*int(self.out_feature_size[index][0]) + j) * self.classes_num + int(obj[0][4]) + 1] = 1

                                    points_label_map[(i*int(self.out_feature_size[index][0]) + j) * 6 + 0] = obj[0][0] # cx
                                    points_label_map[(i*int(self.out_feature_size[index][0]) + j) * 6 + 1] = obj[0][1] # cy
                                    points_label_map[(i*int(self.out_feature_size[index][0]) + j) * 6 + 2] = obj[0][2] # w
                                    points_label_map[(i*int(self.out_feature_size[index][0]) + j) * 6 + 3] = obj[0][3] # h
                                    

                                    points_label_map[(i*int(self.out_feature_size[index][0]) + j) * 6 + 4] = obj[0][5] # difficult
                                    points_label_map[(i*int(self.out_feature_size[index][0]) + j) * 6 + 5] = obj[2] # lamda parameter

                            else:# The point is out inner box.
                                if is_point_in_bbox(ref_point_position, obj[0]):# The point is in object bounding box but out inner box.
                                    class_label_map[(i*int(self.out_feature_size[index][0]) + j) * self.classes_num + 0] = 0
                label_list.append([class_label_map, points_label_map])
        return label_list


# DataLoader中collate_fn使用
def dataset_collate(batch):
    images = []
    targets = []
    targets_cls = []
    targets_loc = []
    bboxes = []
    names = []
    for img, target, box, name in batch:
        images.append(img)
        targets_cls.append(target[0])
        targets_loc.append(target[1])
        bboxes.append(box)
        names.append(name)
    images = np.array(images)
    targets_cls = np.array(targets_cls)
    targets.append(targets_cls)
    targets_loc = np.array(targets_loc)
    targets.append(targets_loc)
    return images, targets, bboxes, names

# DataLoader中collate_fn使用
def dataset_collate_multi_head(batch):
    images = []
    targets = []
    targets_large_cls = []
    targets_large_loc = []
    targets_medium_cls = []
    targets_medium_loc = []
    targets_small_cls = []
    targets_small_loc = []
    bboxes = []
    names = []
    for img, target, box, name in batch:
        images.append(img)

        targets_large_cls.append(target[0][0])
        targets_large_loc.append(target[0][1])

        targets_medium_cls.append(target[1][0])
        targets_medium_loc.append(target[1][1])

        targets_small_cls.append(target[2][0])
        targets_small_loc.append(target[2][1])

        bboxes.append(box)
        names.append(name)
    images = np.array(images)
    targets_large_cls = np.array(targets_large_cls)
    targets_large_loc = np.array(targets_large_loc)
    targets.append([targets_large_cls, targets_large_loc])

    targets_medium_cls = np.array(targets_medium_cls)
    targets_medium_loc = np.array(targets_medium_loc)
    targets.append([targets_medium_cls, targets_medium_loc])

    targets_small_cls = np.array(targets_small_cls)
    targets_small_loc = np.array(targets_small_loc)
    targets.append([targets_small_cls, targets_small_loc])
    return images, targets, bboxes, names

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

def targets_to_labels(targets, output_size, num_classes, bs):
    in_h = output_size[1]
    in_w = output_size[0]

    label_CONF_CLS = targets[0] #bs*in_h,in_w*c  c=num_classes(Include background)
    label_CONF_CLS = label_CONF_CLS.reshape(bs,in_h,in_w,-1) # bs,in_h,in_w,c
    ### bs, in_h, in_w
    label_CONF = np.sum(label_CONF_CLS[:,:,:,1:], axis=3) # bs, in_h, in_w ## Conf
    print(label_CONF[label_CONF>0])

    label_CLS_weight =  np.ceil(label_CONF_CLS) # bs,in_h,in_w,c
    weight_neg = label_CLS_weight[:,:,:,:1] # bs,in_h,in_w,c(c = 1)
    if num_classes > 2:
        weight_non_ignore = np.sum(label_CLS_weight,axis=3).unsqueeze(3)
        weight_pos = (1 - weight_neg)*weight_non_ignore # Exclude rows with all zeros.
    else:
        weight_pos = label_CLS_weight[:,:,:,1:] # bs,in_h,in_w,c(c = 1)

    ### bs,in_h,in_w
    weight_neg = weight_neg.squeeze(3)
    ### bs,in_h,in_w
    weight_pos = weight_pos.squeeze(3)

    
    #################LOC
    label_LOC_difficult_lamda = targets[1] #bs*in_h,in_w*c  c=6(cx,xy,o_w,o_h,difficult,lamda)
    label_LOC_difficult_lamda = label_LOC_difficult_lamda.reshape(bs,in_h,in_w,-1) # bs,in_h,in_w,c(c=6)
    ### bs, in_h, in_w, c(c=4 cx,xy,o_w,o_h)
    label_LOC = label_LOC_difficult_lamda[:,:,:,:4] # bs,in_h,in_w,c(c=4)
    ### bs, in_h, in_w
    label_difficult = label_LOC_difficult_lamda[:,:,:,4] # bs,in_h,in_w
    ### bs, in_h, in_w
    label_lamda = label_LOC_difficult_lamda[:,:,:,5] # bs,in_h,in_w

    return label_CONF, weight_neg, weight_pos, label_LOC, label_difficult, label_lamda

if __name__ == '__main__':
    image_size = (672,384)
    input_mode = "RGB"
    continues_num=5
    assign_method="binary_assign"
    dataset_image_path = "../../../dataset/FlyingBird/train/images/"
    data = open("./img_label_five_continuous_difficulty_train.txt").readlines()
    #################LOC

    stride = 2
    in_w = int(image_size[0]/stride)
    in_h = int(image_size[1]/stride)

    batch_size = 1
    

    train_data = CustomDataset(data, image_size, image_path=dataset_image_path, input_mode=input_mode, continues_num=continues_num, assign_method=assign_method)
    dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=False, collate_fn=dataset_collate)

    print(len(dataloader))
    for i, item in enumerate(dataloader):
        if i > len(dataloader):
            break
        image_datas, targets, bboxes, names = item
        print(names)
        # print(image_datas.size())
        if input_mode == "RGB":
            images = datasetImgTocv2Mat_RGB(image_datas[0])
        else:
            images = datasetImgTocv2Mat_GRG(image_datas[0], continues_num)
    
        label_CONF, weight_neg, weight_pos, label_LOC, label_difficult, label_lamda = targets_to_labels(targets, (in_w, in_h), 2, batch_size)
        label_difficult = label_difficult * weight_pos
        label_lamda = label_lamda * weight_pos
        for bs in range(batch_size):
            test_img = np.full((image_size[1],image_size[0],3), 255)
            test_img = test_img.astype(np.uint8)
            draw_heatmap(test_img, label_CONF[bs], (image_size[1],image_size[0]))
            cv2.imwrite("./test_output_img/heatmapimg_{}.jpg".format(bs), test_img)

            label_points = label_LOC[bs]
            x1 = label_points[..., 0] - label_points[..., 2]/2
            y1 = label_points[..., 1] - label_points[..., 3]/2
            x2 = label_points[..., 0] + label_points[..., 2]/2
            y2 = label_points[..., 1] + label_points[..., 3]/2
            label_points[..., 0] = x1
            label_points[..., 1] = y1
            label_points[..., 2] = x2
            label_points[..., 3] = y2


            label_points = label_points[weight_pos[bs]>0]


            continues_num = len(images)
            # print(continues_num)
            for i, image in enumerate(images):
                if i == int(continues_num/2):
                    if len(label_points) != 0:
                        for box in label_points:
                            # print(box)
                            cv2.rectangle(image,(int(box[0]),int(box[1])),(int(box[2]),int(box[3])),(0,255,0),1)
                            cv2.rectangle(test_img,(int(box[0]),int(box[1])),(int(box[2]),int(box[3])),(0,255,0),1)
                        cv2.imwrite("./test_output_img/label_test{}.jpg".format(i), image)
                        cv2.imwrite("./test_output_img/cls_loc_test{}.jpg".format(bs), test_img)
                    else:
                        cv2.imwrite("./test_output_img/label_test{}.jpg".format(i), image)
                else:
                    cv2.imwrite("./test_output_img/label_test{}.jpg".format(i), image)


        str = input("Enter your input: ")
        if str == "\n":
            continue
        if str == "q":
            break