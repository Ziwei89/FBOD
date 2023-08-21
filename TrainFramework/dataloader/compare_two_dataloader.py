# -*- coding: utf-8 -*-
import sys
sys.path.append("..")
import numpy as np
from torch.utils.data import DataLoader
import dataloader.dataset_bbox as dd_box
import dataloader.mydataset as dm
from utils.getTargets_for_Loss import getTargets

import cv2



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
    input_mode = "GRG"
    continues_num=5
    dataset_image_path = "../../dataset/val/images/"
    data = open("./img_label_five_continuous_difficulty_val.txt").readlines()
    #################LOC

    stride = 2
    in_w = int(image_size[0]/stride)
    in_h = int(image_size[1]/stride)

    batch_size = 2
    

    dataset1 = dd_box.CustomDataset(data, image_size, image_path=dataset_image_path,input_mode=input_mode,continues_num=continues_num)
    dataloader_1 = DataLoader(dataset1, batch_size=batch_size, shuffle=False, collate_fn=dd_box.dataset_collate)

    dataset2 = dm.CustomDataset(data, image_size, image_path=dataset_image_path,input_mode=input_mode, continues_num=continues_num)
    dataloader_2 = DataLoader(dataset2, batch_size=batch_size, shuffle=False)

    get_targets = getTargets(image_size, 2, 0.7, 0.3, 2)

    for item1, item2 in zip(dataloader_1, dataloader_2):
        _, bboxes, names1 = item1
        print("names1:")
        print(names1)
        targets1 = get_targets(batch_size,bboxes)

        _, targets2, names2 = item2
        print("names2:")
        print(names2)
        labels1 = targets1[0]
        labels1=labels1.view(batch_size,in_h,in_w,-1) # bs,in_h,in_w,c
        print("labels1")
        print(labels1)

        loc1 = targets1[1]
        loc1=loc1.view(batch_size,in_h,in_w,-1) # bs,in_h,in_w,c

        labels2 = targets2[0]
        labels2 =labels2.view(batch_size,in_h,in_w,-1) # bs,in_h,in_w,c
        print("labels2")
        print(labels2)

        loc2 = targets2[1]
        loc2=loc2.view(batch_size,in_h,in_w,-1) # bs,in_h,in_w,c

        # for b in range(batch_size):
        #     for i in range(in_h):
        #         for j in range(in_w):
        #             for c in range(2):
        #                 if abs(labels1[b][i][j][c]-labels2[b][i][j][c]) > 0.001:
        #                     print(b,i,j,c,labels1[b][i][j][c],labels2[b][i][j][c])
        # for b in range(batch_size):
        #     for i in range(in_h):
        #         for j in range(in_w):
        #             for c in range(6):
        #                 if abs(loc1[b][i][j][c]-loc2[b][i][j][c]) > 0.001:
        #                     print(b,i,j,c,loc1[b][i][j][c],loc2[b][i][j][c])

        str = input("Enter your input: ")
        if str == "\n":
            continue
        if str == "q":
            break