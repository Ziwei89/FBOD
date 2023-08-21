import os
import torch
import torch.nn as nn
from net.FBODInferenceNet import FBODInferenceBody
from utils.utils import FB_boxdecoder, FBObj
import time


num_to_english_c_dic = {3:"three", 5:"five", 7:"seven", 9:"nine", 11:"eleven"}
class FB_detector(object):
    _defaults = {
        "confidence": 0.3,
        "iou" : 0.3,
        "cuda": True
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"
    #---------------------------------------------------#
    #   Initialize FB_detector
    #---------------------------------------------------#
    ### FBODInferenceBody parameters:
    ### input_img_num=5, aggregation_output_channels=16, aggregation_method="multiinput", input_mode="GRG", ### Aggreagation parameters.
    ### backbone_name="cspdarknet53": ### Extract parameters. input_channels equal to aggregation_output_channels.
    def __init__(self, raw_image_shape=(720,1280), model_input_size=(384,672),
                       input_img_num=5, aggregation_output_channels=16, aggregation_method="multiinput", input_mode="GRG", backbone_name="cspdarknet53", fusion_method="concat",
                       abbr_assign_method="ba", Add_name="0812_1", model_name="FB_object_detect_model.pth",
                       scale=80.):
        self.__dict__.update(self._defaults)

        
        self.raw_image_shape = raw_image_shape
        self.model_input_size=model_input_size

        ############## The parameter transfom the output to raw image
        self.resize_ratio = min(model_input_size[0] / raw_image_shape[0], model_input_size[1] / raw_image_shape[1]) # h,w
        resized_image_shape = (raw_image_shape[0]*self.resize_ratio, raw_image_shape[1]*self.resize_ratio)
        self.offset_top = (model_input_size[0] - resized_image_shape[0])/2
        self.offset_left = (model_input_size[1] - resized_image_shape[1])/2
        ############## The parameter transfom the output to raw image

        # create model
        self.input_img_num = input_img_num
        self.net = FBODInferenceBody(input_img_num=input_img_num, aggregation_output_channels=aggregation_output_channels,
                                     aggregation_method=aggregation_method, input_mode=input_mode, backbone_name=backbone_name, fusion_method=fusion_method).eval()

        # load model
        model_path = "logs/" + num_to_english_c_dic[input_img_num] + "/" + str(model_input_size[0]) + "_" + str(model_input_size[1]) + "/" \
                           + input_mode + "_" + aggregation_method + "_" + backbone_name + "_" + fusion_method + "_" + abbr_assign_method + "_" + Add_name + "/" + model_name
        
        print('Loading weights into state dict...')
        device = torch.device('cuda' if self.cuda else 'cpu')
        state_dict = torch.load(model_path, map_location=device)
        self.net.load_state_dict(state_dict)
        if self.cuda:
            os.environ["CUDA_VISIBLE_DEVICES"] = '0'
            self.net = nn.DataParallel(self.net)
            self.net = self.net.cuda()
        print('{} model loaded.'.format(model_path))

        # initialize boxdecoder
        self.boxdecoder = FB_boxdecoder(model_input_size=model_input_size, score_threshold=self.confidence,
                                        nms_thres=self.iou, scale=scale)
    
    def detect_image(self, images):

        with torch.no_grad():
            images = torch.from_numpy(images)
            if self.cuda:
                images = images.cuda()
            t1 = time.time()
            predictions = self.net(images)
            t2 = time.time()
            print("run model: ", t2-t1)
        outputs = self.boxdecoder(predictions)
        # print("outputs")
        # print(outputs)
        for b in range(len(outputs)):
            outputs[b][:,[0, 2]] = (outputs[b][:,[0, 2]] - self.offset_left) / self.resize_ratio
            outputs[b][:,[1, 3]] = (outputs[b][:,[1, 3]] - self.offset_top) / self.resize_ratio
        return outputs
    
    def inference(self, images):
        with torch.no_grad():
            images = torch.from_numpy(images)
            if self.cuda:
                images = images.cuda()
            predictions = self.net(images)
        return predictions



class FB_Postprocess(object):
    _defaults = {
        "confidence": 0.3,
        "iou" : 0.3,
        "cuda": True
    }
    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"
    
    #---------------------------------------------------#
    #   Initialize FB_postprocess
    #---------------------------------------------------#
    def __init__(self, batch_size, model_input_size=(384,672), raw_image_shape=(720,1280), scale=80.):
        self.__dict__.update(self._defaults)
        self.batch_size = batch_size
        self.model_input_size = model_input_size
        self.raw_image_shape = raw_image_shape

        self.resize_ratio = min(model_input_size[0] / raw_image_shape[0], model_input_size[1] / raw_image_shape[1]) # h,w
        resized_image_shape = (raw_image_shape[0]*self.resize_ratio, raw_image_shape[1]*self.resize_ratio)

        self.offset_top = (model_input_size[0] - resized_image_shape[0])/2
        self.offset_left = (model_input_size[1] - resized_image_shape[1])/2


        self.boxdecoder = FB_boxdecoder(model_input_size=self.model_input_size,
                                        score_threshold=self.confidence, nms_thres=self.iou,
                                        scale=scale)

    def Process(self, model_outputs, iteration):
        obj_result_list = []
        outputs = self.boxdecoder(model_outputs)

        for b in range(len(outputs)):
            outputs[b][:,[0, 2]] = (outputs[b][:,[0, 2]] - self.offset_left) / self.resize_ratio
            outputs[b][:,[1, 3]] = (outputs[b][:,[1, 3]] - self.offset_top) / self.resize_ratio

        for batch_id in range(self.batch_size):
            image_id = self.batch_size*iteration + batch_id
            try:
                batch_detections = outputs[batch_id].cpu().numpy()
            except:
                continue
            # print("batch_detections")
            # print(batch_detections)
            for batch_detection in batch_detections:
                obj_result_list.append(FBObj(score=batch_detection[4], image_id=image_id, bbox=batch_detection[:4]))
        return obj_result_list
