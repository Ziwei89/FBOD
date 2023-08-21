import torch
import torch.nn as nn
import math
import numpy as np
import copy

def boxes_iou(b1, b2):
    """
    输入为：
    ----------
    b1: tensor, shape=(..., 4), xywh
    b2: tensor, shape=(..., 4), xywh

    返回为：
    -------
    iou: tensor, shape=(..., 1)
    """
    # 求出预测框左上角右下角
    b1_xy = b1[..., :2]
    b1_wh = b1[..., 2:4]
    b1_wh_half = b1_wh/2.
    b1_mins = b1_xy - b1_wh_half
    b1_maxes = b1_xy + b1_wh_half
    # 求出真实框左上角右下角
    b2_xy = b2[..., :2]
    b2_wh = b2[..., 2:4]
    b2_wh_half = b2_wh/2.
    b2_mins = b2_xy - b2_wh_half
    b2_maxes = b2_xy + b2_wh_half

    # 求真实框和预测框所有的iou
    intersect_mins = torch.max(b1_mins, b2_mins)
    intersect_maxes = torch.min(b1_maxes, b2_maxes)
    intersect_wh = torch.max(intersect_maxes - intersect_mins, torch.zeros_like(intersect_maxes))
    intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
    b1_area = b1_wh[..., 0] * b1_wh[..., 1]
    b2_area = b2_wh[..., 0] * b2_wh[..., 1]
    union_area = b1_area + b2_area - intersect_area
    iou = intersect_area / torch.clamp(union_area,min = 1e-6)
    return iou

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

class getTargets(nn.Module):
    def __init__(self, model_input_size, num_classes=2, scale=80., stride=2, cuda=True):
        super(getTargets, self).__init__()
        self.model_input_size = model_input_size#(672,384)#img_w,img_h
        self.num_classes = num_classes
        self.scale = scale
        self.out_feature_size = [self.model_input_size[0]/stride, self.model_input_size[1]/stride] ## feature_w,feature_h
        self.size_per_ref_point = self.model_input_size[0]/self.out_feature_size[0]

        self.cuda = cuda

    # def forward(self, batch_size, bboxes_bs):
    def forward(self, input, bboxes_bs):
        # input is a [GHC, CLS, LOC] list with 'bs,c,h,w' format tensor.
        # bboxes is a bs list with 'n,c' tensor, n is the num of box.
        targets = [] ### targets is a list wiht 2 members, each is a 'bs,h,w,c' format tensor(cls and bbox).
        targets_cls = []
        targets_loc = []


        bs = input[0].size(0)
        in_h = input[0].size(2) # in_h = model_input_size[1]/2 (stride = 2)
        in_w = input[0].size(3) # in_w

        ################# Get predict bboxes
        FloatTensor = torch.cuda.FloatTensor if self.cuda else torch.FloatTensor

        # bs, h, w
        ref_point_xs = ((torch.linspace(0, in_w - 1, in_w).repeat(in_h, 1))*(self.model_input_size[0]/in_w) + (self.model_input_size[0]/in_w)/2).repeat(bs, 1, 1)
        ref_point_xs = ref_point_xs.type(FloatTensor)

        # bs, w, h
        ref_point_ys = ((torch.linspace(0, in_h - 1, in_h).repeat(in_w, 1))*(self.model_input_size[1]/in_h) + (self.model_input_size[1]/in_h)/2).repeat(bs, 1, 1)
        # bs, w, h -> bs, h, w
        ref_point_ys = ref_point_ys.permute(0, 2, 1).contiguous()#
        ref_point_ys = ref_point_ys.type(FloatTensor)

        predict_bboxes_bs = input[1] #bs, c,h,w  c=4(dx1,dy1,dx2,dy2)
         # bs, c, h, w -> bs, h, w, c
        predict_bboxes_bs = predict_bboxes_bs.permute(0, 2, 3, 1).contiguous()
        # Decode boxes (x1,y1,x2,y2)
        
        predict_bboxes_bs[..., 0] = predict_bboxes_bs[..., 0]*self.scale + ref_point_xs
        predict_bboxes_bs[..., 1] = predict_bboxes_bs[..., 1]*self.scale + ref_point_ys
        predict_bboxes_bs[..., 2] = predict_bboxes_bs[..., 2]*self.scale + ref_point_xs
        predict_bboxes_bs[..., 3] = predict_bboxes_bs[..., 3]*self.scale + ref_point_ys

        ### bs, h, w, c(c=4)
        ### (x1,y1,x2,y2) ----->  (cx,cy,o_w,o_h)
        predict_bboxes_bs[..., 2] = predict_bboxes_bs[..., 2] - predict_bboxes_bs[..., 0]
        predict_bboxes_bs[..., 3] = predict_bboxes_bs[..., 3] - predict_bboxes_bs[..., 1]
        predict_bboxes_bs[..., 0] = predict_bboxes_bs[..., 0] + predict_bboxes_bs[..., 2]/2
        predict_bboxes_bs[..., 1] = predict_bboxes_bs[..., 1] + predict_bboxes_bs[..., 3]/2
        ###########################

        for b in range(bs):
            bboxes = bboxes_bs[b]
            predict_bboxes = predict_bboxes_bs[b]
            label_list = self.__get_targets_with_dynamicLableAssign(predict_bbox=predict_bboxes,bboxes=bboxes) ### label_list[0], label_list[1] '1,h,w,c'
            targets_cls.append(label_list[0])
            targets_loc.append(label_list[1])
        targets_cls = torch.cat(targets_cls, 0) ### 'bs,h,w,c' format tensor
        targets_loc = torch.cat(targets_loc, 0) ### 'bs,h,w,c' format tensor
        targets.append(targets_cls)
        targets.append(targets_loc)
        return targets
      
    def __get_targets_with_dynamicLableAssign(self, predict_bbox, bboxes): ###

        FloatTensor = torch.cuda.FloatTensor if self.cuda else torch.FloatTensor

        ### predict_bbox feature_h, feature_w, 4 (4: cx, cy, o_w, o_h)
        ### bboxes m,6 (6: x1, y1, x2, y2, class_id, object score)

        label_list=[]
        class_label_map = np.array(([1.] + [0.] * (self.num_classes - 1))*int(self.out_feature_size[0])*int(self.out_feature_size[1])) ### For targets
        points_label_map = np.array([1.]*6*int(self.out_feature_size[0])*int(self.out_feature_size[1])) ### For targets
        if len(bboxes) == 0:

            class_label_map = class_label_map.reshape(1, int(self.out_feature_size[1]), int(self.out_feature_size[0]), -1) ### 1,h,w,c
            points_label_map = points_label_map.reshape(1, int(self.out_feature_size[1]), int(self.out_feature_size[0]), -1) ### 1,h,w,c

            class_label_map = torch.from_numpy(class_label_map)
            points_label_map = torch.from_numpy(points_label_map)
            label_list.append(class_label_map)
            label_list.append(points_label_map)
            return label_list

        position_iou_value_dic = {}

        # convert x1,y1,x2,y2 to cx,cy,o_w,o_h
        ### bboxes m,6 (6:cx, cy, o_w, o_h, class_id, object score)
        bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 0] ## o_w
        bboxes[:, 3] = bboxes[:, 3] - bboxes[:, 1] ## o_h
        bboxes[:, 0] = bboxes[:, 0] + bboxes[:, 2] / 2 # cx
        bboxes[:, 1] = bboxes[:, 1] + bboxes[:, 3] / 2 # cy

        for bbox in bboxes:
            obj_area = bbox[2] * bbox[3]
            if obj_area == 0:
                continue

            ###  bbox[0] cx, bbox[1] cy, bbox[2] o_w, bbox[3] o_h, bbox[4] class_id, bbox[5] difficult ###
            bbox_position = copy.deepcopy(bbox[:4]) ###  bbox_position[0] cx, bbox_position[1] cy, bbox_position[2] o_w, bbox_position[3] o_h
            bbox_position = bbox_position.type(FloatTensor)

            # bbox_position expand 4 to feature_h, feature_w, 4
            ## self.out_feature_size[0] feature_w, self.out_feature_size[1] feature_h
            bbox_position = bbox_position.repeat(int(self.out_feature_size[0]),1).repeat(int(self.out_feature_size[1]),1,1)
            # iou: feature_h, feature_w, 1
            iou = boxes_iou(predict_bbox, bbox_position)
            # iou: feature_h, feature_w
            iou = iou.squeeze()
            if self.cuda:
                iou = iou.cpu().detach().numpy()
            else:
                iou = iou.detach().numpy()
            
            ############### 
            first_filter = np.array([0.]*int(self.out_feature_size[0])*int(self.out_feature_size[1]))
            min_wight_index, min_height_index, max_wight_index, max_height_index = min_max_ref_point_index(bbox,self.out_feature_size,self.model_input_size)
            for i in range(min_height_index, max_height_index+1):
                for j in range(min_wight_index, max_wight_index+1):
                    first_filter[i*int(self.out_feature_size[0]) + j] = 1
            first_filter = first_filter.reshape(int(self.out_feature_size[1]), int(self.out_feature_size[0]))
            
            iou_filter = iou * first_filter  # feature_h, feature_w  # first filter: gt box filter

            dynamic_k = np.sum(iou_filter)
            if dynamic_k < 1:
                dynamic_k = 1
            dynamic_k = math.ceil(dynamic_k)
            dropout_iou = copy.deepcopy(iou_filter)
            dropout_iou = dropout_iou.reshape(-1)
            sorted_iou = np.sort(dropout_iou)
            second_filter_index = iou_filter <= sorted_iou[-(dynamic_k+1)]
            iou_filter[second_filter_index] = 0  # second filter: dynamic_k filter
            ##############
            # The positive anchor points of the object is more, 
            # the weight(lamda) is smaller. To balance the positive anchor points number of different object.
            lamda = (1/dynamic_k)**(1/2)

            for i in range(min_height_index, max_height_index+1):
                for j in range(min_wight_index, max_wight_index+1):
                    if iou_filter[i][j] > 0:
                        if (i,j) in position_iou_value_dic:
                            if iou_filter[i][j] < position_iou_value_dic[(i,j)]:
                                continue
                            elif iou_filter[i][j] == position_iou_value_dic[(i,j)]:
                                for class_id_index in range(self.num_classes): # Ignore this point
                                    class_label_map[(i*int(self.out_feature_size[0]) + j) * self.num_classes + class_id_index] = 0
                                continue
                            else: # iou_filter[i][j] > position_iou_value_dic[(i,j)]
                                position_iou_value_dic[(i,j)] = iou_filter[i][j]

                                for class_id_index in range(self.num_classes): # Reset the class value of this point
                                    class_label_map[(i*int(self.out_feature_size[0]) + j) * self.num_classes + class_id_index] = 0
                        else:
                            position_iou_value_dic[(i,j)] = iou_filter[i][j]
                        
                        class_label_map[(i*int(self.out_feature_size[0]) + j) * self.num_classes + 0] = 0
                        class_label_map[(i*int(self.out_feature_size[0]) + j) * self.num_classes + int(bbox[4]) + 1] = 1

                        points_label_map[(i*int(self.out_feature_size[0]) + j) * 6 + 0] = bbox[0] # cx
                        points_label_map[(i*int(self.out_feature_size[0]) + j) * 6 + 1] = bbox[1] # cy
                        points_label_map[(i*int(self.out_feature_size[0]) + j) * 6 + 2] = bbox[2] # o_w
                        points_label_map[(i*int(self.out_feature_size[0]) + j) * 6 + 3] = bbox[3] # o_h
                        

                        points_label_map[(i*int(self.out_feature_size[0]) + j) * 6 + 4] = bbox[5] # difficult
                        points_label_map[(i*int(self.out_feature_size[0]) + j) * 6 + 5] = lamda # lamda parameter
                    
                    else:# The point is not in positive anchor points.
                        class_label_map[(i*int(self.out_feature_size[0]) + j) * self.num_classes + 0] = 0

        class_label_map = class_label_map.reshape(1, int(self.out_feature_size[1]), int(self.out_feature_size[0]), -1) ### 1,h,w,c
        points_label_map = points_label_map.reshape(1, int(self.out_feature_size[1]), int(self.out_feature_size[0]), -1) ### 1,h,w,c
        class_label_map = torch.from_numpy(class_label_map)
        points_label_map = torch.from_numpy(points_label_map)
        label_list.append(class_label_map)
        label_list.append(points_label_map)
        
        return label_list