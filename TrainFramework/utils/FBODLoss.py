import torch
import torch.nn as nn
import sys
import math
sys.path.append("..")
from .getDynamicTargets import getTargets

def MSELoss(pred,target):
    return (pred-target)**2

def box_ciou(b1, b2):
    """
    输入为：
    ----------
    b1: tensor, shape=(batch, feat_w, feat_h, 4), xywh
    b2: tensor, shape=(batch, feat_w, feat_h, 4), xywh

    返回为：
    -------
    ciou: tensor, shape=(batch, feat_w, feat_h, 1)
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

    # 计算中心的差距
    center_distance = torch.sum(torch.pow((b1_xy - b2_xy), 2), axis=-1)
    
    # 找到包裹两个框的最小框的左上角和右下角
    enclose_mins = torch.min(b1_mins, b2_mins)
    enclose_maxes = torch.max(b1_maxes, b2_maxes)
    enclose_wh = torch.max(enclose_maxes - enclose_mins, torch.zeros_like(intersect_maxes))
    # 计算对角线距离
    enclose_diagonal = torch.sum(torch.pow(enclose_wh,2), axis=-1)
    ciou = iou - 1.0 * (center_distance) / torch.clamp(enclose_diagonal,min = 1e-6)
    
    v = (4 / (math.pi ** 2)) * torch.pow((torch.atan(b1_wh[..., 0]/torch.clamp(b1_wh[..., 1],min = 1e-6)) - torch.atan(b2_wh[..., 0]/torch.clamp(b2_wh[..., 1],min = 1e-6))), 2)
    alpha = v / torch.clamp((1.0 - iou + v),min=1e-6)
    ciou = ciou - alpha * v
    return ciou

class LossFunc(nn.Module): #
    def __init__(self,num_classes, model_input_size=(672,384), scale=80., stride=2, cuda=True,gettargets=False):
        super(LossFunc, self).__init__()
        self.num_classes = num_classes
        self.model_input_size = model_input_size
        self.scale = scale
         #(model_input_size, num_classes=2, stride=2)
        self.get_targets = getTargets(model_input_size, num_classes, scale, stride, cuda)
        self.cuda = cuda
        self.gettargets = gettargets
    
    def forward(self, input, targets):

        FloatTensor = torch.cuda.FloatTensor if self.cuda else torch.FloatTensor
        # targets is bboxes, bbox[0] cx, bbox[1] cy, bbox[2] w, bbox[3] h, bbox[4] class_id, bbox[5] score
        if self.gettargets:
            targets = self.get_targets(input, targets) ### targets is a list wiht 2 members, each is a 'bs,in_h,in_w,c' format tensor(cls and bbox).

        # input is a list with with 2 members(GHC and LOC), each member is a 'bs,c,in_h,in_w' format tensor).
        # print("input[0].size()")
        # print(input[0].size())
        # print("input[1].size()")
        # print(input[1].size())
        bs = input[0].size(0)
        in_h = input[0].size(2) # in_h = model_input_size[1]/stride (stride = 2)
        in_w = input[0].size(3) # in_w

        # 2,bs,c,in_h,in_w -> 2,bs,in_h,in_w,c (a list with 2 members, each member is a 'bs,in_h,in_w,c' format tensor).

        # Branch for task, there are 2 tasks, that is GHC(Gassian Heatmap Conf), and LOC(LOCation).
        # To get 3D tensor 'bs, in_h, in_w' or 4D tensor 'bs, in_h, in_w, c'.
        #################GHC
        # 
        predict_GHC = input[0].type(FloatTensor) #bs,c,in_h,in_w  c=1
        predict_GHC = predict_GHC.view(bs,in_h,in_w) #bs,in_h,in_w
        ### bs, in_h, in_w
        predict_GHC = torch.sigmoid(predict_GHC)


        #################LOC

        # bs, in_h, in_w
        ref_point_xs = ((torch.linspace(0, in_w - 1, in_w).repeat(in_h, 1))*(self.model_input_size[0]/in_w) + (self.model_input_size[0]/in_w)/2).repeat(bs, 1, 1)
        ref_point_xs = ref_point_xs.type(FloatTensor)

        # bs, in_w, in_h
        ref_point_ys = ((torch.linspace(0, in_h - 1, in_h).repeat(in_w, 1))*(self.model_input_size[1]/in_h) + (self.model_input_size[1]/in_h)/2).repeat(bs, 1, 1)
        # bs, in_w, in_h -> bs, in_h, in_w
        ref_point_ys = ref_point_ys.permute(0, 2, 1).contiguous()#
        ref_point_ys = ref_point_ys.type(FloatTensor)

        predict_LOC = input[1].type(FloatTensor) #bs, c,in_h,in_w  c=4(dx1,dy1,dx2,dy2)
        # bs, c, in_h, in_w -> bs, in_h, in_w, c
        predict_LOC = predict_LOC.permute(0, 2, 3, 1).contiguous()
        # Decode boxes (x1,y1,x2,y2)
        
        predict_LOC[..., 0] = predict_LOC[..., 0]*self.scale + ref_point_xs
        predict_LOC[..., 1] = predict_LOC[..., 1]*self.scale + ref_point_ys
        predict_LOC[..., 2] = predict_LOC[..., 2]*self.scale + ref_point_xs
        predict_LOC[..., 3] = predict_LOC[..., 3]*self.scale + ref_point_ys

        ### bs, in_h, in_w, c(c=4)
        ### (x1,y1,x2,y2) ----->  (cx,cy,o_w,o_h)
        predict_LOC[..., 2] = predict_LOC[..., 2] - predict_LOC[..., 0]
        predict_LOC[..., 3] = predict_LOC[..., 3] - predict_LOC[..., 1]
        predict_LOC[..., 0] = predict_LOC[..., 0] + predict_LOC[..., 2]/2
        predict_LOC[..., 1] = predict_LOC[..., 1] + predict_LOC[..., 3]/2
        ###########################

        # targets is a list wiht 2 members, each is a 'bs*in_h,in_w*c' format tensor(cls and bbox).
        # 2,bs*c*in_h,in_w -> 3,bs,in_h,in_w,c (a list with 3 members, each member is a 'bs,in_h,in_w,c' format tensor).

        #################GHC_CLS
        ### bs, in_h, in_w, c(c=num_classes(Include background))
        label_GHC_CLS = targets[0].type(FloatTensor) #bs*in_h,in_w*c  c=num_classes(Include background)
        label_GHC_CLS = label_GHC_CLS.view(bs,in_h,in_w,-1) # bs,in_h,in_w,c
        ### bs, in_h, in_w
        # print("label_GHC_CLS[:,:,:,1:].size()")
        # print(label_GHC_CLS[:,:,:,1:].size())
        label_GHC = torch.sum(label_GHC_CLS[:,:,:,1:], dim=3) # bs, in_h, in_w ## Guassian Heat Conf
        # print("label_GHC.size()")
        # print(label_GHC.size())

        label_CLS_weight =  torch.ceil(label_GHC_CLS) # bs,in_h,in_w,c
        weight_neg = label_CLS_weight[:,:,:,:1] # bs,in_h,in_w,c(c = 1)
        if self.num_classes > 2:
            weight_non_ignore = torch.sum(label_CLS_weight,3).unsqueeze(3)
            weight_pos = (1 - weight_neg)*weight_non_ignore # Exclude rows with all zeros.
        else:
            weight_pos = label_CLS_weight[:,:,:,1:] # bs,in_h,in_w,c(c = 1)

        ### bs,in_h,in_w
        weight_neg = weight_neg.squeeze(3)
        ### bs,in_h,in_w
        weight_pos = weight_pos.squeeze(3)
        ### bs
        bs_neg_nums = torch.sum(weight_neg, dim=(1,2))
        ### bs
        bs_obj_nums = torch.sum(weight_pos, dim=(1,2))
        
        #################LOC
        label_LOC_difficult_lamda = targets[1].type(FloatTensor) #bs*in_h,in_w*c  c=6(cx,xy,o_w,o_h,difficult,lamda)
        label_LOC_difficult_lamda = label_LOC_difficult_lamda.view(bs,in_h,in_w,-1) # bs,in_h,in_w,c(c=6)
        ### bs, in_h, in_w, c(c=4 cx,xy,o_w,o_h)
        label_LOC = label_LOC_difficult_lamda[:,:,:,:4] # bs,in_h,in_w,c(c=4)
        ### bs, in_h, in_w
        # label_difficult = label_LOC_difficult_lamda[:,:,:,4] # bs,in_h,in_w
        ### bs, in_h, in_w
        label_lamda = label_LOC_difficult_lamda[:,:,:,5] # bs,in_h,in_w

        ## Guassian Conf Loss
        ## bs, in_h, in_w
        # print("predict_GHC[predict_GHC>0.2]")
        # print(predict_GHC[predict_GHC>0.2])
        MSE_Loss = MSELoss(label_GHC, predict_GHC)
        neg_MSE_Loss = MSE_Loss * weight_neg
        pos_MSE_Loss = (MSE_Loss * label_lamda) * weight_pos

        GHC_loss = 0
        for b in range(bs):
            GHC_loss_per_batch = 0
            ### in_h, in_w
            if bs_obj_nums[b] != 0:
                k = bs_obj_nums[b].cpu()
                k = int(k.numpy())
                topk = 2*k
                if topk > bs_neg_nums[b]:
                    topk = bs_neg_nums[b]
                neg_MSE_Loss_topk_sum = torch.sum(torch.topk((neg_MSE_Loss[b]).view(-1), topk).values)
                pos_MSE_Loss_sum = torch.sum(pos_MSE_Loss[b])
                GHC_loss_per_batch = (neg_MSE_Loss_topk_sum + 10*pos_MSE_Loss_sum)/bs_obj_nums[b]
            else:
                neg_MSE_Loss_topk_sum = torch.sum(torch.topk(neg_MSE_Loss[b], 20).values)
                GHC_loss_per_batch = neg_MSE_Loss_topk_sum/10
            GHC_loss += GHC_loss_per_batch
        
        ### Locate Loss
        ciou_loss = 1-box_ciou(predict_LOC, label_LOC)
        ###(bs, in_h, in_w)
        ciou_loss = (ciou_loss.view(bs,in_h,in_w)) * label_lamda * weight_pos
        LOC_loss = 0
        for b in range(bs):
            LOC_loss_per_batch = 0
            if bs_obj_nums[b] != 0:
                LOC_loss_per_batch = torch.sum(ciou_loss[b])/bs_obj_nums[b]
            else:
                LOC_loss_per_batch = 0
            LOC_loss += LOC_loss_per_batch

        total_loss = (10*GHC_loss + 100*LOC_loss) / bs
        return total_loss