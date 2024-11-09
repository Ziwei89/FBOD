[简体中文](README.md)  
# FBOD-SV
[A Flying Bird Object Detection method for Surveillance Video ](https://ieeexplore.ieee.org/document/10614237)   
Aiming at the specific characteristics of flying bird  objects in surveillance video, such as the typically non-obvious  features in single-frame images, small size in most instances,  and asymmetric shapes, this article proposes a flying bird object  detection method for surveillance video (FBOD-SV). Firstly,  a new feature aggregation module, the correlation attention  feature aggregation (Co-Attention-FA) module, is designed to  aggregate the features of the flying bird object according to  the bird object’s correlation on multiple consecutive frames  of images. Secondly, a flying bird object detection network (FBOD-Net) with down-sampling followed by up-sampling is  designed, which utilizes a large feature layer that fuses fine  spatial information and large receptive field information to detect  special multiscale (mostly small scale) bird objects. Finally,  the SimOTA dynamic label allocation method is applied to  one-category object detection, and the SimOTA-OC dynamic  label strategy is proposed to solve the difficult problem of  label allocation caused by irregular flying bird objects. The article overview diagram is shown below.  
<div align="center">
  <img src="https://github.com/Ziwei89/FBOD/blob/master/Illustrative_Figure/framework_github.jpeg">
</div>


**<font color=red>Note: </font>This project is mainly aimed at the objects whose features are not obvious in a single frame and whose size is small in the vast majority of cases (occasionally large object). This type of dataset is used to train FBOD-Net, which has the ability to detect large objects. If the FBOD-Net is trained using a dataset where most cases are large objects, the training process may be difficult to converge.**

**<font color=red>Note: </font>The dataset used in this article (surveillance video flying bird object dataset), part of which cannot be made public. We have collected the publicly available dataset FBD-SV-2024, which is available! Please visit https://github.com/Ziwei89/FBD-SV-2024_github for  the dataset download link and related processing scripts.**

This project is the article implementation code. The model input of this project is n consecutive frames of images (take 5 consecutive frames as an example), and the bounding box information of the flying bird object in the middle frame is predicted (if n=1, the bounding box information of the flying bird object in the current frame is predicted.    
<font color=red>Note: </font>A small amount of code was sourced from other projects or the Internet. Please contact sun_zi_wei@my.swjtu.edu.cn for special instructions or rights requirements. This project can be used for study and research, not for commercial activities.  

<font color=red>Please refer to the paper if the project is useful for your work: </font>
```
@ARTICLE{2024_FBOD-SV,
  author={Sun, Zi-Wei and Hua, Ze-Xi and Li, Heng-Chao and Li, Yan},
  journal={IEEE Transactions on Instrumentation and Measurement}, 
  title={A Flying Bird Object Detection Method for Surveillance Video}, 
  year={2024},
  volume={73},
  number={},
  pages={1-14},
  doi={10.1109/TIM.2024.3435183}}
```


# Project Application Steps

## 1. Clone the project
```
git clone https://github.com/Ziwei89/FBOD.git
```
## 2. The main environment of the project (the main environment of the author's runtime)
ubuntu  
python=3.10.6  
numpy=1.23.4  
pytorch=1.11.0=py3.10_cuda11.3_cudnn8.2.0_0  
python-opencv=4.6.0  
tqdm=4.64.1  
imgaug=0.4.0  
matplotlib=3.6.2  

## 3. Prepare training and test data
**<font color=red>Note: </font> By default, the project root directory (FBOD/) is the starting place to work before running the script.**

You can use labelImg to annotate the image and get an xml file.  
The label file should contain the object bounding box, the object category, and the difficulty information of the object (we will use the difficulty information of the object in a future project, so don't worry about it in this project, and the relevant code will set a default value that is not needed).  

### (1) Dataset organization
```  
data_root_path/  
               videos/
                     train/
                           bird_1.mp4
                           bird_2.mp4
                           ...  
                     val/
                images/
                     train/  
                           bird_1_000000.jpg  
                           bird_1_000001.jpg  
                           ...  
                           bird_2_000000.jpg  
                           ...  
                     val/
                labels
                     train/  
                           bird_1_000000.xml  
                           bird_1_000001.xml  
                           ...
                           bird_2_000000.xml
                           ...  
                     val/
```  
### (2) Generate data description txt files for training and testing (one txt file for training and one txt file for testing).
The format of the data description txt file is as follows:  
The first frame image name of n consecutive frames *space* The bird object information of the middle frame  
image_name x1,y1,x2,y2,cls,difficulty x1,y1,x2,y2,cls,difficulty  
eg:  
```
...  
bird_3_000143.jpg 995,393,1016,454,0,0.625
bird_3_000144.jpg 481,372,489,389,0,0.375 993,390,1013,456,0,0.625
...  
bird_40_000097.jpg None
...
```
We provide a script that generates such a data description txt file. This script is provided as continuous_image_annotation.py in the Data_process/ directory, and to run it we need to specify the path to the dataset and the number of consecutive image frames to input to the model for one inference:  
(The script continuous_image_annotation_frames_padding.py adds the sequence padding, which adds black images before and after the beginning and end of the video to produce output for the first and last frames; see our article for details.)  
```
cd Data_process # Go to the data processing directory from the project root.
python continuous_image_annotation.py \
       --data_root_path=../dataset/FBD-SV-2024/ \
       --input_img_num=5
```
After running the script, two txt files will be generated in the TrainFramework/dataloader/ directory. The two txt files are img_label_five_continuous_difficulty_train_raw.txt and img_label_five_continuous_difficulty_val_raw.txt. The training samples in these two files are in order, so it's best to shuffle them by running the following script:  
```
cd TrainFramework/dataloader/ # Go to the dataloader directory from the project root.
python shuffle_txt_lines.py \
       --input_img_num=5
```
After running this script, two txt files ( img_label_five_continuous_difficulty_train.txt and img_label_five_continuous_difficulty_val.txt) will be generated in the TrainFramework/dataloader/ directory.  
### (3) Prepare the classes txt file
Create a model_data folder inside the TrainFramework/ directory, and then create a file called classes.txt inside the TrainFramework/model_data/ directory with the classes, like this:  
```
bird
```

## 4. Train the bird object detection model
When training the model, we need to use command-line arguments. All can be set parameters are defined in TrainFramework/config/opts.py files (including training and testing the required parameters), these parameters have default values, can not set.  
  Some of the parameters are explained as follows (the other parameters are more general, so refer to TrainFramework/config/opts.py when setting them):  
```
input_img_num                      # Number of consecutive video frames fed to the model at one time
input_mode                         # Input video frame image color mode, provide two modes to choose: RGB, GRG. One is that all frames are RGB, the other is that only the middle frame is RGB, while the other frames are grayscale images
aggregation_method                 # The way of feature aggregation for n consecutive frames of images. There are three options: relatedatten, multiinput, and convlstm. Where relatedatten represents relevant attention aggregation, multiinput represents input after concat of multiple frames, and convlstm represents aggregation using convlstm manner
aggregation_output_channels        # The number of output channels after feature aggregation of n consecutive frames of video images (which can be appropriately adjusted according to the number of input consecutive frames)
fusion_method                      # Fusion mode of shallow feature layer and deep feature layer
assign_method                      # Label assignment method, the project provides 3 methods, shrink bounding box (binary_assign, ba), center Gaussian (guassian_assign, ga), SimOTA-OC(auto_assign, aa)
scale_factor                       # The object scale normalization factor of the single-scale output model, which is used to normalize the coordinate data of the object to a smaller range
scale_min_max_list                 # Multi-scale (3 scales) output model maximum-minimum sequence of object scales. The objects scale range is counted, and it is divided into large scale, medium scale and small scale targets, and their maximum and minimum sizes were recorded respectively. Used to normalize the coordinate data of the object to a smaller range
pretrain_model_path                # The path of the pre-trained model. When using auto_assign label assignment method, using static label assignment method for pre-training will achieve better results
Add_name                           # Add a suffix to the relevant record files (such as the model save folder or the training record image)
data_augmentation                  # Whether to use data augmentation at training time
start_Epoch                        # The starting training Epoch, which is usually set when the trained model is loaded and continues training, can be used to adjust the learning rate to continue training
```
**<font color=red>Note: </font>Here, there are five training scripts, including "label_assign_in_dataloader" suffix for static label assignment (assignment in dataloader), When using the contracted bounding box (binary_assign) or centered Gaussian (guassian_assign) label assignment, select the training script with this suffix. When selecting the SimOTA-OC(auto_assign) assignment, select training scripts that do not contain this suffix.**
A training script with "multi_scale" indicates that the output of the trained model is multi-scale (3 scales). Training scripts with the "_with_absolute_path" suffix indicate that when training, in the label description script, the absolute path of the image is specified.  
An example of training:  
```
cd TrainFramework # Go to the training framewrok from the project root
python train_AP50.py \
        --model_input_size=384_672 \ # Model input image size，h_w
        --input_img_num=5 \
        --input_mode=RGB \
        --aggregation_method=relatedatten \
        --backbone_name=cspdarknet53 \
        --fusion_method=concat \
        --scale_factor=80 \
        --assign_method=auto_assign \
        --pretrain_model_path=logs/five/384_672/RGB_relatedatten_cspdarknet53_concat_ga_20230822/FB_object_detect_model.pth \
        --Batch_size=8 \
        --data_root_path=../dataset/FBD-SV-2024/ \
        --data_augmentation=True \
        --Add_name=20230822
```
During training, the program will create a directory logs/five/384_672/RGB_relatedatten_cspdarknet53_concat_aa_20230822/ inside the TrainFramework/ directory that will hold some of the trained models. Also, a loss.jpg image will be created in this directory to record the loss during training, and an AP50.jpg image will be created after 30 epochs to record the ap50 performance metrics of subsequent models.  

## 5. Testing and evaluating the flying bird object detection model
When running the test and evaluation scripts, make sure that the command-line arguments are the same as those used during training (you don't need to include training-specific arguments such as Batch_size; you need to specifically add arguments for the model name (because there are multiple models in the model folder)), or the script will fail to find the model.  
A few examples corresponding to the above training (all on the test set):  

* Test the images in the test set (n consecutive frames of image input), the red box represents the detection results, the green box represents GT:  
```
cd TrainFramework # Go to the training framewrok from the project root
python predict_for_image.py \
        --model_input_size=384_672 \
        --input_img_num=5 \
        --input_mode=RGB \
        --aggregation_method=relatedatten \
        --backbone_name=cspdarknet53 \
        --fusion_method=concat \
        --scale_factor=80 \
        --assign_method=auto_assign \
        --data_root_path=../dataset/FBD-SV-2024/ \
        --Add_name=20230822 \
        --model_name=Epoch80-Total_Loss6.4944-Val_Loss16.7809-AP_50_0.7611.pthc
```
The output images are in the TrainFramework/test_output/ directory. During the run, the terminal output prompts you to choose whether to continue testing the next image (other keys than key "q") or to exit the test (key "q").   

* Detecting videos in the test set (note: The video to be detected is stored in the directory $data_root_path/videos/val/), the red box represents the detection result, and the green box represents GT (the script with the suffix "frames_padding" means that the sequence padding technology is used to make the output for the first and last frames of the video. Please refer to our article):  
```
cd TrainFramework # Go to the training framewrok from the project root
python predict_for_video.py \
        --model_input_size=384_672 \
        --input_img_num=5 \
        --input_mode=RGB \
        --aggregation_method=relatedatten \
        --backbone_name=cspdarknet53 \
        --fusion_method=concat \
        --scale_factor=80 \
        --assign_method=auto_assign \
        --data_root_path=../dataset/FBD-SV-2024/ \
        --Add_name=20230822 \
        --model_name=FB_object_detect_model.pth \
        --video_name=bird_2.mp4
```
The output video is in the TrainFramework/test_output/ directory.  

* Detect the given video (note: the full path of the video needs to be given), the red box shows the detection result: 
```
cd TrainFramework # Go to the training framewrok from the project root
python predict_for_video_with_video_full_path.py \
        --model_input_size=384_672 \
        --input_img_num=5 \
        --input_mode=RGB \
        --aggregation_method=relatedatten \
        --backbone_name=cspdarknet53 \
        --fusion_method=concat \
        --scale_factor=80 \
        --assign_method=auto_assign \
        --Add_name=20230822 \
        --model_name=FB_object_detect_model.pth \
        --video_full_path=./test.mp4
```
The output video is in the TrainFramework/test_output/ directory. The difference between predict_for_video.py and predict_for_video_with_video_full_path.py is that the former will detect the videos in the dataset (you specify the name of the video in the test set) and have label information, while the second will be a user-specified video (you need to give all the paths).  

* Model evaluation (running all test videos) (The script with the suffix "frames_padding" means that the sequential padding technique is used, so that the first and last frames of the video have output results and the AP will be improved, please refer to our article for details):  
```
cd TrainFramework # Go to the training framewrok from the project root
python mAP_for_AllVideo_coco_tools.py \
        --model_input_size=384_672 \
        --input_img_num=5 \
        --input_mode=RGB \
        --aggregation_method=relatedatten \
        --backbone_name=cspdarknet53 \
        --fusion_method=concat \
        --scale_factor=80 \
        --assign_method=auto_assign \
        --data_root_path=../dataset/FBD-SV-2024/ \
        --Add_name=20230822 \
        --model_name=Epoch80-Total_Loss6.4944-Val_Loss16.7809-AP_50_0.7611.pth
```
* Model evaluation (run a single video, test set of videos):  
```
cd TrainFramework # Go to the training framewrok from the project root
python mAP_for_video.py \
        --model_input_size=384_672 \
        --input_img_num=5 \
        --input_mode=RGB \
        --aggregation_method=relatedatten \
        --backbone_name=cspdarknet53 \
        --fusion_method=concat \
        --scale_factor=80 \
        --assign_method=auto_assign \
        --data_root_path=../dataset/FBD-SV-2024/ \
        --Add_name=20230822 \
        --model_name=Epoch80-Total_Loss6.4944-Val_Loss16.7809-AP_50_0.7611.pth \
        --video_name=bird_7.mp4
```
Count the model's detections (count the number of detections on the test set (total_bbox), the number of correct detections (True_detection), the number of missed detections (False_background), and the number of false detections (False_detection)):  
```
cd TrainFramework # Go to the training framewrok from the project root
python count_detection_info.py \
        --model_input_size=384_672 \
        --input_img_num=5 \
        --input_mode=RGB \
        --aggregation_method=relatedatten \
        --backbone_name=cspdarknet53 \
        --fusion_method=concat \
        --scale_factor=80 \
        --assign_method=auto_assign \
        --data_root_path=../dataset/FBD-SV-2024/ \
        --Add_name=20230822 \
        --model_name=Epoch80-Total_Loss6.4944-Val_Loss16.7809-AP_50_0.7611.pth
```