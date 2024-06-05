import pycocotools.coco as coco
from pycocotools.cocoeval import COCOeval
import json

from FB_detector import FB_detector
import os
import cv2
import numpy as np
from PIL import Image
from queue import Queue
from utils.common import GetMiddleImg_ModelInput
import xml.etree.ElementTree as ET
from config.opts import opts
from utils.utils import FBObj

class label_FBObj():
    def __init__(self, image_id=False, score_list=None, class_id_list=False, bbox_list=None):
        self.image_id = image_id
        self.score_list = score_list
        self.class_id_list = class_id_list
        self.bbox_list = bbox_list


classes=['bird']
def Convert_Annotation_coco_Label(annotation_file, image_id):
    image_id = image_id
    score_list = []
    class_id_list = []
    bbox_list = []
    in_file = open(annotation_file, encoding='utf-8')
    tree=ET.parse(in_file)
    root = tree.getroot()

    for obj in root.iter('object'):
        cls = obj.find('name').text
        if cls not in classes:
            continue
        score_list.append(1.0)
        cls_id = classes.index(cls)
        class_id_list.append(int(cls_id))
        xmlbox = obj.find('bndbox')
        bbox = [int(xmlbox.find('xmin').text), int(xmlbox.find('ymin').text), int(xmlbox.find('xmax').text), int(xmlbox.find('ymax').text)]
        bbox_list.append(bbox)
    label_obj = label_FBObj(image_id=image_id, score_list=score_list, class_id_list=class_id_list, bbox_list=bbox_list)
    return label_obj


def movingobj_to_coco_label(label_obj_list, label_json_file, width=1280, height=720):
    image_info = []
    categories = [{"supercategory": "bird", "id": 0, "name": "bird"}]
    annotations = []
    ann_id = 1
    for label_obj in label_obj_list:
        image_name = str(label_obj.image_id)
        info = {
            "file_name": image_name,
            "height": height,
            "width": width,
            "id": label_obj.image_id,
        }
        image_info.append(info)
        for box, class_id in zip(label_obj.bbox_list, label_obj.class_id_list):
            xmin = int(box[0])
            ymin = int(box[1])
            xmax = int(box[2])
            ymax = int(box[3])
            w = xmax - xmin
            h = ymax - ymin
            coco_box = [max(xmin, 0), max(ymin, 0), min(w, width), min(h, height)]
            ann = {
                "image_id": label_obj.image_id,
                "bbox": coco_box,
                "category_id": class_id,
                "iscrowd": 0,
                "id": ann_id,
                "area": coco_box[2] * coco_box[3],
            }
            annotations.append(ann)
            ann_id += 1
    coco_dict = {
        "images": image_info,
        "categories": categories,
        "annotations": annotations,
    }
    json_fp = open(label_json_file, 'w')
    json_str = json.dumps(coco_dict)
    json_fp.write(json_str)
    json_fp.close()

if __name__ == "__main__":

    image_total_id = 0
    all_label_obj_list = []
    all_obj_result_list = []

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
                              abbr_assign_method=abbr_assign_method, Add_name=Add_name, model_name=model_name, scale=opt.scale_factor)


    label_path = opt.data_root_path + "val/labels/" #.xlm label file path

    video_path = opt.data_root_path + "val/video/"

    continus_num = input_img_num
    
    image_total_id = 0
    all_label_obj_list = []
    all_obj_result_list = []
    video_names = os.listdir(video_path)
    label_name_list=os.listdir(label_path)

    for video_name in video_names:
        image_q = Queue(maxsize=continus_num)
        start_image_total_id = image_total_id
        
        cap=cv2.VideoCapture(video_path + video_name)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        ################# frames padding ################
        for i in range(int(continus_num/2)):
            black_image = np.zeros((height, width, 3), dtype=np.uint8)
            image_q.put(black_image)
        #################################################

        frame_id = 0
        while (True):
            ret,frame=cap.read()
            if ret != True:
                break
            else:
                image_total_id += 1
                frame_id += 1
                image = Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
                image_q.put(image)
                image_shape = np.array(np.shape(image)[0:2]) # image size is 1280,720; image array's shape is 720,1280
                img_width = int(image_shape[1])
                img_heigth = int(image_shape[0])
                # print("image_shape:")
                # print(image_shape)
                if frame_id >= int(continus_num/2) + 1 and frame_id <= frame_count:
                    exist_label = False

                    ### The output of model is start from 1st frame now
                    frame_id_str = "%06d" % int((frame_id-1)-int(continus_num/2)) #The frame id in dataset start from 0, but this script start from 1.
                    label_name = video_name.split(".")[0] + "_" + frame_id_str + ".xml"
                    if label_name in label_name_list:
                        exist_label = True
                        all_label_obj_list.append(Convert_Annotation_coco_Label(label_path + label_name, start_image_total_id + (frame_id-int(continus_num/2))))
                    
                    # If there's no label for the middle frame of this input quene, continue this detection.
                    if exist_label == False:
                        _ = image_q.get()
                        continue

                    _, model_input = GetMiddleImg_ModelInput(image_q, model_input_size=model_input_size, continus_num=continus_num, input_mode=input_mode)
                    _ = image_q.get()
                    outputs = fb_detector.detect_image(model_input, raw_image_shape=image_shape)

                    obj_result_list = []
                    for output in outputs[0]: ###
                        # print(output)
                        box = [0,0,0,0]
                        box[0] = output[0].item()
                        box[1] = output[1].item()
                        box[2] = output[2].item()
                        box[3] = output[3].item()
                        # print("predict:")
                        # print(box)
                        score = output[4].item()
                        ### The output of detector is delayed by int(continus_num/2) frames.
                        obj_result_list.append(FBObj(score=score, image_id=start_image_total_id + (frame_id-int(continus_num/2)), bbox=box))
                    all_obj_result_list += obj_result_list
                if frame_id == frame_count: ## Output the detection results of the last int(continus_num/2) frames of the video.
                    for n in range(1, int(continus_num/2)+1):
                        
                        black_image = np.zeros((height, width, 3), dtype=np.uint8)
                        image_q.put(black_image)

                        exist_label = False
                        frame_id_str = "%06d" % int((frame_id-1) - (int(continus_num/2)-n)) #The frame id in dataset start from 0, but this script start from 1.
                        label_name = video_name.split(".")[0] + "_" + frame_id_str + ".xml"
                        if label_name in label_name_list:
                            exist_label = True
                            all_label_obj_list.append(Convert_Annotation_coco_Label(label_path + label_name, start_image_total_id + (frame_id-(int(continus_num/2)-n))))
                        
                        # If there's no label for the middle frame of this input quene, continue this detection.
                        if exist_label == False:
                            _ = image_q.get()
                            continue
                        _, model_input = GetMiddleImg_ModelInput(image_q, model_input_size=model_input_size, continus_num=continus_num, input_mode=input_mode)
                        _ = image_q.get()
                        outputs = fb_detector.detect_image(model_input, raw_image_shape=image_shape)

                        obj_result_list = []
                        for output in outputs[0]: ###
                            # print(output)
                            box = [0,0,0,0]
                            box[0] = output[0].item()
                            box[1] = output[1].item()
                            box[2] = output[2].item()
                            box[3] = output[3].item()
                            # print("predict:")
                            # print(box)
                            score = output[4].item()
                            ### The output of detector is delayed by int(continus_num/2) frames.
                            obj_result_list.append(FBObj(score=score, image_id=start_image_total_id + (frame_id-(int(continus_num/2)-n)), bbox=box))
                        all_obj_result_list += obj_result_list
    
    label_json_file = "./instances_test2017.json"
    movingobj_to_coco_label(all_label_obj_list, label_json_file, width=img_width, height=img_heigth)
    
    predict_data = []
    for obj_result in all_obj_result_list:
        obj_dic = {}
        obj_dic["image_id"] = int(obj_result.image_id)
        obj_dic["category_id"] = int(0)
        box = obj_result.bbox
        xmin = float(box[0])
        ymin = float(box[1])
        xmax = float(box[2])
        ymax = float(box[3])
        w = xmax - xmin
        h = ymax - ymin
        coco_box = [max(xmin, 0), max(ymin, 0), min(w, img_width), min(h, img_heigth)]
        obj_dic["bbox"] = coco_box
        obj_dic["score"] = float(obj_result.score)
        predict_data.append(obj_dic)
    
    predict_json_data = json.dumps(predict_data)
    with open('results_test.json', 'w') as f:
        f.write(predict_json_data)
    
    
    
    results = "./results_test.json" ##模型预测结果
    anno = "./instances_test2017.json"  ##ground truth
    coco_anno = coco.COCO(anno)
    coco_dets = coco_anno.loadRes(results)
    coco_eval = COCOeval(coco_anno, coco_dets, "bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()