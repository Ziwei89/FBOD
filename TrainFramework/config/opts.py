import argparse

class opts(object):
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        
        self.parser.add_argument('--model_input_size', default="384_672", type=str,
                            help='model_input_size: The model input shape h_w')
        
        self.parser.add_argument('--input_img_num', default=5, type=int,
                            help='input_img_num: Number of consecutive video frames fed to the model at one time')
        
        self.parser.add_argument('--input_mode', default="RGB", type=str,
                            help='input_mode: "RGB" or "GRG". "RGB": mean RGB ..RGB, "GRG": mean GRAY ..RGB.. GRAY')
        
        self.parser.add_argument('--aggregation_method', default='multiinput', type=str,
                            help='aggregation_method: multiinput, relatedatten or convlstm')
        
        self.parser.add_argument('--aggregation_output_channels', default=16, type=int,
                            help='aggregation_output_channels: The output channels of the aggregation module')
        
        self.parser.add_argument('--backbone_name', default='cspdarknet53', type=str,
                            help='backbone_name: cspdarknet53 or CustomNet_model')
        
        self.parser.add_argument('--fusion_method', default='concat', type=str,
                            help='fusion_method: concat or scm')
        
        self.parser.add_argument('--assign_method', default='auto_assign', type=str,
                            help='assign_method: The label assign method. binary_assign, guassian_assign or auto_assign')
        
        self.parser.add_argument('--Add_name', default='0816_1', type=str,
                            help='Add_name: add name to logs and pic')
        
        self.parser.add_argument('--data_root_path', default="../../dataset/FBD-SV-2024/", type=str,
                            help='data_root_path: The path of the dataset.')
        
        self.parser.add_argument('--scale_factor', default=80, type=int,
                            help='scale_factor: Object scale normalization factor for single scale output model.')
        
        self.parser.add_argument('--scale_min_max_list', default="80,256,48,80,13,48", type=str,
                            help='scale_min_max_list: Object scale normalization factor for multi scale output model, "s1,s2,s3,s4,s5,s6", "s1,s2" is the max min value of large scale. \
                                  Need to count the object scale, and divide to 3 sets, log the min max of each set.')

        ######### for train
        self.parser.add_argument('--Batch_size', default=8, type=int,
                            help='Batch_size: The size of batch.')
        
        self.parser.add_argument('--data_augmentation', default=True, type=bool,
                            help='data_augmentation: Determin whether to augmentate the dataset.')
        
        self.parser.add_argument('--pretrain_model_path', default="logs/non.pth", type=str,
                            help='pretrain_model_path: the pretrain model to speed train.')
        
        self.parser.add_argument('--lr', default=1e-3, type=float,
                            help='lr: the learn rate.')
        
        self.parser.add_argument('--start_Epoch', default=0, type=int,
                            help='start_Epoch: the start epoch.')
        
        self.parser.add_argument('--end_Epoch', default=100, type=int,
                            help='start_Epoch: the end epoch.')
        
        self.parser.add_argument('--cross_vx', default="cross_v1", type=str,
                            help='cross_vx: The Cross Validation data set')
        
        ######### for test
        self.parser.add_argument('--model_name', default="FB_object_detect_model.pth", type=str,
                            help='model_name: The model name for loade.')
        
        self.parser.add_argument('--video_name', default="bird_2.mp4", type=str,
                            help='video_name: The vido name for testing.')
        
        self.parser.add_argument('--video_full_path', default="./test.mp4", type=str,
                            help='video_full_path: input the video with full path for predict.')
        
    def parse(self):
        opt = self.parser.parse_args()
        return opt