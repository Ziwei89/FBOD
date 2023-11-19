# -*- coding: utf-8 -*-
import os
import sys
sys.path.append("..")
import numpy as np
from torch.utils.data import Dataset, DataLoader
import cv2
import dataloader.augmentations as DataAug
from PIL import Image


class CustomDataset(Dataset):
    def __init__(self, train_lines, image_size, image_path, input_mode="GRG", continues_num=5, data_augmentation=False):
        # input_mode: "RGB" or "GRG". "RGB" means all the image is rgb mode. "GRG" means that the middle image remains RGB,
        # and the others will be coverted to gray.
        self.train_lines = train_lines
        self.train_batches = len(train_lines)
        self.image_size = image_size#(960,544)
        self.image_path = image_path
        self.input_mode = input_mode
        self.frame_num = continues_num
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
            images = DataAug.Resize((self.image_size[1], self.image_size[0]), False)(np.copy(images), np.copy(bboxes))
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
        return images, bboxes

    def __getitem__(self, index):
        lines = self.train_lines
        img_list, y = self.__load_data(lines[index])
        if self.input_mode == "RGB":
            img_inp = self.__Cv2ToImage_Concate(img_list)
        else:
            img_inp = self.__Cv2ToImage_OnlyMidRGB_Concate(img_list)
        targets = y
        return img_inp, targets, lines[index].split(".")[0]

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
    
# DataLoader中collate_fn使用
def dataset_collate(batch):
    images = []
    bboxes = []
    names = []
    for img, box, name in batch:
        images.append(img)
        bboxes.append(box)
        names.append(name)
    images = np.array(images)
    return images, bboxes, names



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

    batch_size = 2
    

    train_data = CustomDataset(data, image_size, image_path=dataset_image_path,input_mode=input_mode,continues_num=continues_num)
    dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=False, collate_fn=dataset_collate)
    print(len(dataloader))
    for i, item in enumerate(dataloader):
        if i > len(dataloader):
            break
        image_datas, targets, names = item
        print(names)
        # print(image_datas.size())
        for b in range(batch_size):
            if input_mode == "RGB":
                images = datasetImgTocv2Mat_RGB(image_datas[b])
            else:
                images = datasetImgTocv2Mat_GRG(image_datas[b], continues_num)
            
            write_img = images[int(continues_num/2)]
            bboxes = targets[b]
            for box in bboxes:
                cv2.rectangle(write_img,(int(box[0]),int(box[1])),(int(box[2]),int(box[3])),(0,255,0),2)#x1,y1,x2,y2
            cv2.imwrite("./test_output_img/test_{}.png".format(b), write_img)

        str = input("Enter your input: ")
        if str == "\n":
            continue
        if str == "q":
            break