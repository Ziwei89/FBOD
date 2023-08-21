import os
import cv2
import numpy as np
import copy
from PIL import Image
import dataloader.augmentations as DataAug

def letterbox_image(image, size):
    iw, ih = image.size
    w, h = size
    scale = min(w/iw, h/ih)
    nw = int(iw*scale)
    nh = int(ih*scale)

    image = image.resize((nw,nh), Image.BICUBIC)
    new_image = Image.new('RGB', size, (128,128,128))
    new_image.paste(image, ((w-nw)//2, (h-nh)//2))
    return new_image

def list_allisNone(listobj):
    for obj in listobj:
        if obj is not None:
            return False
    return True

def load_data(line, image_path, frame_num, image_size):
    """line of train_lines was saved as 'image name, label'"""
    line =  line.split()
    first_img_name = line[0]
    first_img_num_str = first_img_name.split(".")[0].split("_")[-1]
    first_img_num = int(first_img_num_str)
    images = []
    for num in range(first_img_num, first_img_num + frame_num):
        num_str = "%05d" % int(num)
        img_name = first_img_name.split(first_img_num_str)[0] + num_str + ".jpg"
        image_full_name = os.path.join(image_path,img_name)
        image = cv2.imread(image_full_name)
        images.append(image)
    if  line[1:][0] == "None":
        bboxes = np.array([])
        # images = DataAug.Resize(image_size, False)(np.copy(images), np.copy(bboxes))
    else:
        bboxes = np.array([np.array(list(map(float, box.split(',')))) for box in line[1:]])
        # images, bboxes = DataAug.Resize(image_size, True)(np.copy(images), np.copy(bboxes))
    return images, bboxes, first_img_name

# np.set_printoptions(threshold=np.inf)
# This function is different from obj detection stage.
def CropImageList_Im2Cv(image_list, position):
    # position is [x1, y1, x2, y2]
    crop_image_list = []
    for image in image_list:
        opencv_image = cv2.cvtColor(np.asarray(image),cv2.COLOR_RGB2BGR) 
        crop_image = copy.copy(opencv_image[position[1]:position[3],position[0]:position[2]])
        crop_image_list.append(crop_image)
    return crop_image_list

def GetMiddleImg_ModelInput(image_quene, model_input_size=(544,960), continus_num=5, input_mode="GRG"):
    # input_mode: "GRG" or "RGB".
    input_list = []
    img_list = image_quene.queue
    MiddleImg = copy.copy(img_list[int(continus_num/2)])# Obtain the Middle frame.
    if input_mode == "RGB":
        for img in img_list:# Differen't from second stage.
            reshaped_img = letterbox_image(img, (model_input_size[1],model_input_size[0]))
            img = np.array(reshaped_img, dtype=np.float32)# img shape is h,w,c
            img /= 255.0
            img = np.transpose(img, (2, 0, 1))
            img = img.astype(np.float32)
            input_list.append(img)
    elif input_mode == "GRG":
        for i, img in enumerate(img_list):
            reshaped_img = letterbox_image(img, (model_input_size[1],model_input_size[0]))
            if i == int(continus_num/2):
                img = np.array(reshaped_img, dtype=np.float32)# img shape is h,w,c
            else:
                img = np.array(reshaped_img.convert('L'), dtype=np.float32)
                img = img.reshape(model_input_size[0],model_input_size[1],1) # h,w,1
            img /= 255.0
            img = np.transpose(img, (2, 0, 1))
            img = img.astype(np.float32)
            input_list.append(img)
    else:
        raise print("Error! input_mode error.")
    inputs = np.concatenate(input_list, 0)
    inputs = inputs.reshape(1,-1,model_input_size[0],model_input_size[1])
    return MiddleImg, inputs

def GetMiddleImg_ModelInput_for_MatImageList(img_list, model_input_size=(544,960), continus_num=5, input_mode="GRG"):
    # input_mode: "GRG" or "RGB".
    input_list = []
    MiddleImg = copy.copy(img_list[int(continus_num/2)])# Obtain the Middle frame.
    if input_mode == "RGB":
        for img in img_list:# Differen't from second stage.
            img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            reshaped_img = letterbox_image(img, (model_input_size[1],model_input_size[0]))
            img = np.array(reshaped_img, dtype=np.float32)# img shape is h,w,c
            img /= 255.0
            img = np.transpose(img, (2, 0, 1))
            img = img.astype(np.float32)
            input_list.append(img)
    elif input_mode == "GRG":
        for i, img in enumerate(img_list):
            img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            reshaped_img = letterbox_image(img, (model_input_size[1],model_input_size[0]))
            if i == int(continus_num/2):
                img = np.array(reshaped_img, dtype=np.float32)# img shape is h,w,c
            else:
                img = np.array(reshaped_img.convert('L'), dtype=np.float32)
                img = img.reshape(model_input_size[0],model_input_size[1],1) # h,w,1
            img /= 255.0
            img = np.transpose(img, (2, 0, 1))
            img = img.astype(np.float32)
            input_list.append(img)
    else:
        raise print("Error! input_mode error.")
    inputs = np.concatenate(input_list, 0)
    inputs = inputs.reshape(1,-1,model_input_size[0],model_input_size[1])
    return MiddleImg, inputs

def correct_boxes(top, left, bottom, right, input_shape, image_shape):
    new_shape = image_shape*np.min(input_shape/image_shape)

    offset = (input_shape-new_shape)/2./input_shape
    scale = input_shape/new_shape

    box_yx = np.concatenate(((top+bottom)/2,(left+right)/2),axis=-1)/input_shape
    box_hw = np.concatenate((bottom-top,right-left),axis=-1)/input_shape

    box_yx = (box_yx - offset) * scale
    box_hw *= scale

    box_mins = box_yx - (box_hw / 2.)
    box_maxes = box_yx + (box_hw / 2.)
    boxes =  np.concatenate([
        box_mins[:, 0:1],
        box_mins[:, 1:2],
        box_maxes[:, 0:1],
        box_maxes[:, 1:2]
    ],axis=-1)
    boxes *= np.concatenate([image_shape, image_shape],axis=-1)
    return boxes