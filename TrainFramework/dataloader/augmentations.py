# coding=utf-8
import cv2
import random
import numpy as np
import imgaug.augmenters as iaa


class HSV(object):
    def __init__(self, hgain=0.015, sgain=0.7, vgain=0.4, p=0.75):
        self.hgain = hgain
        self.sgain = sgain
        self.vgain = vgain
        self.p = p
    def __call__(self, imgs, bboxes):
        if random.random() < self.p:
            x = np.random.uniform(-1, 1, 3) * [self.hgain, self.sgain, self.vgain] + 1
            images = []
            for img in imgs:
                img_hsv = (cv2.cvtColor(img, cv2.COLOR_BGR2HSV) * x).clip(None, 255).astype(np.uint8)
                np.clip(img_hsv[:, :, 0], None, 179, out=img_hsv[:, :, 0])
                img = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR, dst=img)
                images.append(img)
        else:
            images = imgs
        return images, bboxes

class Noise(object):
    def __init__(self, intensity=0.01, p=0.15):
        self.intensity = intensity
        self.p = p
    def __call__(self, imgs, bboxes):
        if random.random() < self.p:
            noise_aug = iaa.AdditiveGaussianNoise(scale=(0, self.intensity * 255))
            images = []
            for img in imgs:
                img = noise_aug.augment_image(img)
                images.append(img)
        else:
            images = imgs
        return images, bboxes

class GuassianNoise(object):
    def __init__(self, intensity=0.05, p=0.2):
        self.intensity = intensity
        self.p = p
    def __call__(self, imgs, bboxes):
        if random.random() < self.p:
            h_img, w_img, _ = imgs[0].shape
            noise = np.random.normal(0, self.intensity, (h_img, w_img, 3)) * 255
            noise = np.array(noise, dtype=np.uint8)
            images = []
            for img in imgs:
                img += noise
                img = np.clip(img, 0, 255)
                images.append(img)
        else:
            images = imgs
        return images, bboxes

class RandomVerticalFilp(object):
    def __init__(self, p=0.5):
        self.p = p
    def __call__(self, imgs, bboxes):
        if random.random() < self.p:
            h_img, _, _ = imgs[0].shape
            images = []
            for img in imgs:
                img = img[::-1, :, :]
                images.append(img)
            if len(bboxes) == 0:
                bboxes = bboxes
            else:
                bboxes[:, 1], bboxes[:, 3] = h_img - bboxes[:, 3], h_img - bboxes[:, 1]
        else:
            images = imgs
        return images, bboxes

class RandomHorizontalFilp(object):
    def __init__(self, p=0.5):
        self.p = p
    def __call__(self, imgs, bboxes):
        if random.random() < self.p:
            _, w_img, _ = imgs[0].shape
            images = []
            for img in imgs:
                img = img[:, ::-1, :]
                images.append(img)
            if len(bboxes) == 0:
                bboxes = bboxes
            else:
                bboxes[:, 0], bboxes[:, 2] = w_img - bboxes[:, 2], w_img - bboxes[:, 0]

        else:
            images = imgs
        return images, bboxes

class RandomCenterFilp(object):
    def __init__(self, p=0.5):
        self.p = p
    def __call__(self, imgs, bboxes):
        if random.random() < self.p:
            h_img, w_img, _ = imgs[0].shape
            images = []
            for img in imgs:
                img = img[::-1, ::-1, :]
                images.append(img)
            if len(bboxes) == 0:
                bboxes = bboxes
            else:
                bboxes[:, 0], bboxes[:, 2] = w_img - bboxes[:, 2], w_img - bboxes[:, 0]
                bboxes[:, 1], bboxes[:, 3] = h_img - bboxes[:, 3], h_img - bboxes[:, 1]
        else:
            images = imgs
        return images, bboxes

class RandomCrop(object):
    def __init__(self, p=0.3):
        self.p = p

    def __call__(self, imgs, bboxes):
        if random.random() < self.p:
            h_img, w_img, _ = imgs[0].shape
            if len(bboxes) == 0:
                max_bbox = [int(w_img/4), int(h_img/4), int(3*w_img/4), int(3*h_img/4)]
            else:
                max_bbox = [np.min(bboxes[:, 0]), np.min(bboxes[:, 1]), np.max(bboxes[:, 2]), np.max(bboxes[:, 3])]
            max_l_trans = max_bbox[0]
            max_u_trans = max_bbox[1]
            max_r_trans = w_img - max_bbox[2]
            max_d_trans = h_img - max_bbox[3]
            crop_xmin = max(0, int(max_bbox[0] - random.uniform(0, max_l_trans)))
            crop_ymin = max(0, int(max_bbox[1] - random.uniform(0, max_u_trans)))
            crop_xmax = min(w_img, int(max_bbox[2] + random.uniform(0, max_r_trans)))#
            crop_ymax = min(h_img, int(max_bbox[3] + random.uniform(0, max_d_trans)))#
            images=[]
            for img in imgs:
                img = img[crop_ymin : crop_ymax, crop_xmin : crop_xmax]
                images.append(img)
            if len(bboxes) == 0:
                bboxes = bboxes
            else:
                bboxes[:, [0, 2]] = bboxes[:, [0, 2]] - crop_xmin
                bboxes[:, [1, 3]] = bboxes[:, [1, 3]] - crop_ymin
        else:
            images = imgs
        return images, bboxes

class Resize(object):

    def __init__(self, target_shape, correct_box=True):
        self.h_target, self.w_target = target_shape
        self.correct_box = correct_box

    def __call__(self, imgs, bboxes):
        h_org , w_org , _= imgs[0].shape

        resize_ratio = min(1.0 * self.w_target / w_org, 1.0 * self.h_target / h_org)
        resize_w = int(resize_ratio * w_org)
        resize_h = int(resize_ratio * h_org)

        images=[]
        for img in imgs:
            image_resized = cv2.resize(img, (resize_w, resize_h))
            dh = self.h_target - resize_h
            top = int(dh/2)
            bottom = dh - top

            dw = self.w_target - resize_w
            left = int(dw/2)
            right = dw - left

            image_paded = cv2.copyMakeBorder(image_resized,top,bottom,left,right,cv2.BORDER_CONSTANT,value=[128,128,128])
            image = image_paded
            images.append(image)

        if self.correct_box:
            ################################xmin-ymax trans
            if len(bboxes) == 0:
                bboxes = bboxes
            else:
                bboxes[:, [0, 2]] = bboxes[:, [0, 2]] * resize_ratio + left
                bboxes[:, [1, 3]] = bboxes[:, [1, 3]] * resize_ratio + top
            return images, bboxes
        return images

class Resize_padingBR(object):

    def __init__(self, target_shape, correct_box=True):
        self.h_target, self.w_target = target_shape
        self.correct_box = correct_box

    def __call__(self, imgs, bboxes):
        h_org , w_org , _= imgs[0].shape
        

        resize_ratio = min(1.0 * self.w_target / w_org, 1.0 * self.h_target / h_org)
        resize_w = int(resize_ratio * w_org)
        resize_h = int(resize_ratio * h_org)

        images = []
        for img in imgs:
            image_resized = cv2.resize(img, (resize_w, resize_h))
            bottom = self.h_target - resize_h
            right = self.w_target - resize_w
            image_paded = cv2.copyMakeBorder(image_resized,0,bottom,0,right,cv2.BORDER_CONSTANT,value=[128,128,128])
            image = image_paded
            images.append(image)

        if self.correct_box:
            ################################xmin-ymax trans
            if len(bboxes) == 0:
                bboxes = bboxes
            else:
                bboxes[:, [0, 2]] = bboxes[:, [0, 2]] * resize_ratio
                bboxes[:, [1, 3]] = bboxes[:, [1, 3]] * resize_ratio
            return images, bboxes
        return images