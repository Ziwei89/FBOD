# FBOD-SV
[A Flying Bird Object Detection method for Surveillance Video ](https://ieeexplore.ieee.org/document/10614237)
基于监控视频飞鸟特点的监控视频飞鸟目标检测方法  
该论文对监控视频中飞鸟目标存在单帧图像特征不明显、大多数情况下尺寸较小以及非对称规则等特征，提出了一种监控视频飞鸟目标检测方法。首先，设计了一种新的特征聚合模块，相关注意力特征聚合模块(Co-Attention-FA)，依据飞鸟目标在连续多帧图像上的相关关系，对飞鸟目标的特征进行聚合。其次，设计了一种先下采样，再上采样的飞鸟目标检测网络(FBOD-Net)，利用一个融合了细腻的空间信息与大感受野信息的大特征层来检测特殊的多尺度(大多数情况为小尺度)飞鸟目标。最后，简化了SimOTA动态标签分配方法，提出了SimOTA-OC动态标签分配策略，解决了因飞鸟目标不规则而导致的标签分配困难问题。论文概况图如下所示。  
<div align="center">
  <img src="https://github.com/Ziwei89/FBOD/blob/master/Illustrative_Figure/framework_github.jpeg">
</div>


**<font color=red>注意：</font>该项目主要针对的是单帧图像特征不明显、绝大多数情况尺寸较小的目标(偶有大目标)。采用这种类型的数据集训练FBOD-Net，具备检测大目标的能力。如采用大多数情况为大目标的数据集训练FBOD-Net，训练过程可能难以收敛。**

**<font color=red>注意：</font>论文中使用的数据集(监控视频飞鸟目标数据)，部分不能公开。我们已经采集可以公开的数据FBD-SV-2024，将于近期公布。敬请期待！**

本项目是该论文实现代码。本项目模型输入是连续n帧图像(以连续5帧为例)，预测飞鸟目标在中间帧的边界框信息(如果n=1, 则预测飞鸟目标在当前帧的边界框信息)  
<font color=red>注意：</font>项目中少量代码来源于其他工程或互联网，在此鸣谢。若需要特别指明或存在权力要求，请联系sun_zi_wei@my.swjtu.edu.cn。本项目可以用于学习和科研，不可用于商业行为。

# 项目应用步骤

## 1、克隆项目到本地
```
git clone https://github.com/Ziwei89/FBOD.git
```
## 2、准备训练和测试数据

可以使用labelImg对图片进行标注，得到xml文件。  
标签文件中应包含目标边界框、目标类别以及目标的难度信息(在以后的项目中会用到目标的难度信息，在本项目中标注时不用考虑，相关代码会设置一个用不到的默认值)

### (1) 数据组织
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
### (2) 生成数据描述txt文件用于训练和测试(训练一个txt,测试一个txt)
数据描述txt文件的格式如下：  
连续n帧图像的第一帧图像名字 *空格* 中间帧飞鸟目标信息  
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
我们提供了一个脚本，可以生成这样的数据描述txt。该脚本为Data_process目录下的continuous_image_annotation.py (脚本continuous_image_annotation_frames_padding.py增加了序列padding, 序列padding就是在视频的开头前和结尾后增加一些全黑的图片，使前几帧和后几帧有输出结果，具体请参考我们的论文)，运行该脚本需要指定数据集路径以及模型一次推理所输入的连续图像帧数：  
```
cd Data_process
python continuous_image_annotation.py \
       --data_root_path=../dataset/FlyingBird/ \
       --input_img_num=5
cd ../ #回到项目根目录
```
运行该脚本后，将在TrainFramework/dataloader/目录下生成两个txt文件，分别是img_label_five_continuous_difficulty_train_raw.txt和img_label_five_continuous_difficulty_val_raw.txt文件。这两个文件中的训练样本排列是顺序的，最好通过运行以下脚本将其打乱：  
```
cd TrainFramework/dataloader/
python shuffle_txt_lines.py \
       --input_img_num=5
cd ../../ #回到项目根目录
```
运行该脚本后，将在TrainFramework/dataloader/目录下生成img_label_five_continuous_difficulty_train.txt和img_label_five_continuous_difficulty_val.txt两个文件。
### (3) 准备类别txt文件
在TrainFramework/目录下创建model_data文件夹，然后再在TrainFramework/model_data/目录下创建名为classes.txt的文件，该文件中记录类别,如：
```
bird
```

## 3、训练飞鸟目标检测模型
在模型训练时，需要使用命令行参数形式进行传参。所有可设置参数均定义在TrainFramework/config/opts.py文件中(包括训练和测试所需要的参数)，这些参数均有默认值，可以不设置。  
部分参数解释如下(其他参数较为通用，在设置时请参考TrainFramework/config/opts.py文件)：  
```
input_img_num                      #一次输入模型中，连续视频图像的帧数
input_mode                         #输入视频帧图像的颜色模式，提供两种模式，一种是所有帧均为RGB，另一种仅仅中间帧为RGB，而其他帧为灰度图像
aggregation_method                 #连续n帧图像特征聚合方式
aggregation_output_channels        #连续n帧视频图像经过特征聚合后输出的通道数(可以根据输入连续帧数适当进行调整)
fusion_method                      #浅层特征层和深层特征层融合方式
assign_method                      #标签分配方式，项目提供3种方式，和收缩边界框(binary_assign, ba)、中心高斯(guassian_assign, ga)、SimOTA-OC(auto_assign, aa)
scale_factor                       #单尺度输出模型，目标尺度归一化因子，用于将目标的坐标数据归一化到一个较小的范围
scale_min_max_list                 #多尺度(3个尺度)输出模型，目标尺度最大最小数列。统计目标尺度范围，将其划分为大尺度、中等尺度、小尺度目标，分别记录其最大最小尺寸。用于将目标的坐标数据归一化到一个较小的范围
pretrain_model_path                #预训练模型的路径。在使用auto_assign标签分配方式时，先采用静态的标签分配方式进行预训练，会取得更好的效果
Add_name                           #在相关记录文件(如模型保存文件夹或训练记录图片)，增加后缀
data_augmentation                  #是否在训练时使用数据增强
start_Epoch                        #起始训练Epoch，一般用在加载训练过的模型继续训练时设置，它可以调整学习率以便接着训练
```
**<font color=red>注意：</font>这里训练脚本有5个，其中有“label_assign_in_dataloader”后缀的表示静态标签分配方式(在dataloader中进行分配)，使用收缩边界框(binary_assign)、中心高斯(guassian_assign)标签分配方式时，应选择含有该后缀的训练脚本，选择SimOTA-OC(auto_assign)分配方式时，应选择不含有该后缀的训练脚本。**
含有“multi_scale”的训练脚本表示训练的模型输出是多尺度的(3个尺度)。有“_with_absolute_path”后缀的训练脚本表示训练时，标签描述脚本中，图像指定绝对路径。
训练的一个例子:  
```
cd TrainFramework
python train_AP50.py \
        --model_input_size=384_672 \
        --input_img_num=5 \
        --input_mode=RGB \
        --aggregation_method=relatedatten \
        --backbone_name=cspdarknet53 \
        --fusion_method=concat \
        --scale_factor=80 \
        --assign_method=auto_assign \
        --pretrain_model_path=logs/five/384_672/RGB_relatedatten_cspdarknet53_concat_ga_20230822/FB_object_detect_model.pth \
        --Batch_size=8 \
        --data_root_path=../dataset/FlyingBird/ \
        --data_augmentation=True \
        --Add_name=20230822
cd ../ #回到项目根目录
```
训练时，程序将在TrainFramework/目录下创建一个logs/five/384_672/RGB_relatedatten_cspdarknet53_concat_aa_20230822/的目录，该目录会保存一些训练模型。同时，会创建一张train_output_img/five/384_672/RGB_relatedatten_cspdarknet53_concat_aa_20230822_loss.jpg的图像用于记录训练过程的loss，训练到30个epoch后，还会创建一张train_output_img/five/384_672/RGB_relatedatten_cspdarknet53_concat_aa_20230822_ap50.jpg的图像用于记录后续模型的AP50性能指标。  

## 4、测试和评价飞鸟目标检测模型
在运行测试和评价脚本时，要保持命令行参数与训练时一致(不用包括训练特有的如Batch_size等参数，需要特别增加模型名字的参数(因为保存模型的文件夹下有多个模型))，否则脚本将报无法找到模型的错误。
几个与上述训练对应的例子(均是运行测试集中的数据)：

* 测试检测测试集中的图片(连续n帧图片输入)，红框表示检测结果，绿色框表示GT：  
```
cd TrainFramework
python predict_for_image.py \
        --model_input_size=384_672 \
        --input_img_num=5 \
        --input_mode=RGB \
        --aggregation_method=relatedatten \
        --backbone_name=cspdarknet53 \
        --fusion_method=concat \
        --scale_factor=80 \
        --assign_method=auto_assign \
        --data_root_path=../dataset/FlyingBird/ \
        --Add_name=20230822 \
        --model_name=Epoch80-Total_Loss6.4944-Val_Loss16.7809-AP_50_0.7611.pth
cd ../ #回到项目根目录
```
输出图片在TrainFramework/test_output/目录下。 运行过程中，通过终端输出提示，是否继续测试下一张图像(除按键q外的其他按键)，或者退出测试(按键q)。 

* 测试检测测试集中的视频(注意：待检测视频存放在$data_root_path/val/video/目录下)，红框表示检测结果，绿色框表示GT (带有后缀_frames_padding的脚本表示采用序列padding技术，使前几帧和后几帧有输出结果，具体请参考我们的论文)： 
```
cd TrainFramework
python predict_for_video.py \
        --model_input_size=384_672 \
        --input_img_num=5 \
        --input_mode=RGB \
        --aggregation_method=relatedatten \
        --backbone_name=cspdarknet53 \
        --fusion_method=concat \
        --scale_factor=80 \
        --assign_method=auto_assign \
        --data_root_path=../dataset/FlyingBird/ \
        --Add_name=20230822 \
        --model_name=FB_object_detect_model.pth \
        --video_name=bird_2.mp4
cd ../ #回到项目根目录
```
输出视频在TrainFramework/test_output/目录下。 

* 测试检测给定的视频(注意：需要给视频的全路径)，红框表示检测结果： 
```
cd TrainFramework
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
cd ../ #回到项目根目录
```
输出视频在TrainFramework/test_output/目录下。predict_for_video.py和predict_for_video_with_video_full_path.py的区别是，前者检测的数据集中的视频(指定测试集中视频名字即可)，有标签信息，后者是用户指定的视频(需要给全部路径)。

* 模型评价(运行所有测试视频) (带有后缀_frames_padding的脚本表示采用序列padding技术，使前几帧和后几帧有输出结果，AP会有所提升，具体请参考我们的论文)：
```
cd TrainFramework
python mAP_for_AllVideo_coco_tools.py \
        --model_input_size=384_672 \
        --input_img_num=5 \
        --input_mode=RGB \
        --aggregation_method=relatedatten \
        --backbone_name=cspdarknet53 \
        --fusion_method=concat \
        --scale_factor=80 \
        --assign_method=auto_assign \
        --data_root_path=../dataset/FlyingBird/ \
        --Add_name=20230822 \
        --model_name=Epoch80-Total_Loss6.4944-Val_Loss16.7809-AP_50_0.7611.pth
cd ../ #回到项目根目录
```
* 模型评价(运行单个视频)：
```
cd TrainFramework
python mAP_for_video.py \
        --model_input_size=384_672 \
        --input_img_num=5 \
        --input_mode=RGB \
        --aggregation_method=relatedatten \
        --backbone_name=cspdarknet53 \
        --fusion_method=concat \
        --scale_factor=80 \
        --assign_method=auto_assign \
        --data_root_path=../dataset/FlyingBird/ \
        --Add_name=20230822 \
        --model_name=Epoch80-Total_Loss6.4944-Val_Loss16.7809-AP_50_0.7611.pth \
        --video_name=bird_2.mp4
cd ../ #回到项目根目录
```
