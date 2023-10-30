import torch
from collections import Counter

from FB_detector import FB_detector
import numpy as np
from utils.common import load_data, GetMiddleImg_ModelInput_for_MatImageList
from utils.utils import FBObj
from config.opts import opts



def IOU(box1, box2):
    """
        计算IOU
    """
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]

    inter_rect_x1 = max(b1_x1, b2_x1)
    inter_rect_y1 = max(b1_y1, b2_y1)
    inter_rect_x2 = min(b1_x2, b2_x2)
    inter_rect_y2 = min(b1_y2, b2_y2)

    inter_area = max(inter_rect_x2 - inter_rect_x1 + 1, 0) * \
                 max(inter_rect_y2 - inter_rect_y1 + 1, 0)
                 
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)

    return iou

def mean_average_precision(pred_oriented_objs,true_oriented_objs,iou_threshold):

    epsilon=1e-6#防止分母为0
    detections = pred_oriented_objs
    ground_truths = true_oriented_objs
    #img 0 has 3 bboxes
    #img 1 has 5 bboxes
    #就像这样：amount_bboxes={0:3,1:5}
    #统计每一张图片中真实框的个数,train_idx指示了图片的编号以区分每张图片
    amount_bboxes=Counter(gt.image_id for gt in ground_truths)

    for key,val in amount_bboxes.items():
        amount_bboxes[key]=torch.zeros(val)#置0，表示这些真实框初始时都没有与任何预测框匹配
    #此时，amount_bboxes={0:torch.tensor([0,0,0]),1:torch.tensor([0,0,0,0,0])}

    #将预测框按照置信度从大到小排序
    detections.sort(key=lambda x:x.score,reverse=True)

    #初始化TP,FP
    TP=torch.zeros(len(detections))
    FP=torch.zeros(len(detections))

        #TP+FN就是当前类别GT框的总数，是固定的
    total_true_bboxes=len(ground_truths)
    
    #如果一个GT框都没有，那么直接返回
    if total_true_bboxes == 0:
        return 0, 0, 0

    #对于每个预测框，先找到它所在图片中的所有真实框，然后计算预测框与每一个真实框之间的IoU，大于IoU阈值且该真实框没有与其他预测框匹配，则置该预测框的预测结果为TP，否则为FP
    for detection_idx,detection in enumerate(detections):
        #在计算IoU时，只能是同一张图片内的框做，不同图片之间不能做
        #图片的编号存在第0个维度
        #于是下面这句代码的作用是：找到当前预测框detection所在图片中的所有真实框，用于计算IoU
        ground_truth_img=[bbox for bbox in ground_truths if bbox.image_id==detection.image_id]

        best_iou=0
        for idx,gt in enumerate(ground_truth_img):
            #计算当前预测框detection与它所在图片内的每一个真实框的IoU
            iou=IOU(detection.bbox,gt.bbox)
            if iou >best_iou:
                best_iou=iou
                best_gt_idx=idx
        if best_iou>iou_threshold:
            #这里的detection[0]是amount_bboxes的一个key，表示图片的编号，best_gt_idx是该key对应的value中真实框的下标
            if amount_bboxes[detection.image_id][best_gt_idx]==0:#只有没被占用的真实框才能用，0表示未被占用（占用：该真实框与某预测框匹配【两者IoU大于设定的IoU阈值】）
                TP[detection_idx]=1#该预测框为TP
                amount_bboxes[detection.image_id][best_gt_idx]=1#将该真实框标记为已经用过了，不能再用于其他预测框。因为一个预测框最多只能对应一个真实框（最多：IoU小于IoU阈值时，预测框没有对应的真实框)
            else:
                FP[detection_idx]=1#虽然该预测框与真实框中的一个框之间的IoU大于IoU阈值，但是这个真实框已经与其他预测框匹配，因此该预测框为FP
        else:
            FP[detection_idx]=1#该预测框与真实框中的每一个框之间的IoU都小于IoU阈值，因此该预测框直接为FP
    TP_cumsum=torch.cumsum(TP,dim=0)
    FP_cumsum=torch.cumsum(FP,dim=0)

    TP_sum = torch.sum(TP)
    FP_sum = torch.sum(FP)
    
    #套公式
    recalls=TP_cumsum/(total_true_bboxes+epsilon)
    precisions=torch.divide(TP_cumsum,(TP_cumsum+FP_cumsum+epsilon))

    recalls_value=TP_sum/(total_true_bboxes+epsilon)
    precisions_value=TP_sum/(TP_sum+FP_sum+epsilon)

    #把[0,1]这个点加入其中
    precisions=torch.cat((torch.tensor([1]),precisions))
    recalls=torch.cat((torch.tensor([0]),recalls))
    #使用trapz计算AP
    average_precision_value = torch.trapz(precisions,recalls)

    return average_precision_value, recalls_value, precisions_value

def labels_to_results(bboxes, image_id):
    label_obj_list = []
    for bbox in bboxes:
        label_obj_list.append(FBObj(score=1.0, image_id=image_id, bbox=bbox[:4]))
    return label_obj_list

num_to_english_c_dic = {1:"one", 3:"three", 5:"five", 7:"seven", 9:"nine", 11:"eleven"}

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
    
    annotation_path = "./dataloader/" + "img_label_" + num_to_english_c_dic[input_img_num] + "_continuous_difficulty.txt"
    dataset_image_path = opt.data_path + "val/images/"

    # val_lines = open(annotation_path).readlines()
    # # # 0.1用于验证，0.9用于训练
    val_split = 0.1
    with open(annotation_path) as f:
        lines = f.readlines()
    np.random.seed(10101)
    np.random.shuffle(lines)
    np.random.seed(None)
    num_val = int(len(lines)*val_split) 
    num_train = len(lines) - num_val

    train_lines = lines[:num_train]
    val_lines = lines[num_train:]

    all_label_obj_list = []
    all_obj_result_list = []
    for i, line in enumerate(val_lines):
        images, bboxes, _ = load_data(line, dataset_image_path, frame_num=input_img_num)
        raw_image_shape = np.array(images[0].shape[0:2]) # h,w
        all_label_obj_list += labels_to_results(bboxes, i)

        _, model_input= GetMiddleImg_ModelInput_for_MatImageList(images, model_input_size=model_input_size, continus_num=input_img_num, input_mode=input_mode)
        outputs = fb_detector.detect_image(model_input, raw_image_shape=raw_image_shape)

        if outputs != None:
            obj_result_list = []
            for output in outputs:
                box = output[:4]
                score = output[4]
                obj_result_list.append(FBObj(score=score, image_id=i, bbox=box))
            all_obj_result_list += obj_result_list
    AP_50,REC_50,PRE_50=mean_average_precision(all_obj_result_list,all_label_obj_list,iou_threshold=0.5) # Include background
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