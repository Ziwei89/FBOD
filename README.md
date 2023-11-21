# FBOD
Flying bird object detection in surveillance video method based on the characteristics of surveillance video bird  
基于监控视频飞鸟特点的监控视频飞鸟目标检测方法  
该论文对监控视频中飞鸟目标存在单帧图像特征不明显、大多数情况下尺寸较小以及非对称规则等特征，提出了一种监控视频飞鸟目标检测方法。首先，设计了一种新的特征聚合模块，相关注意力特征聚合模块，依据飞鸟目标在连续多帧图像上的相关关系，对飞鸟目标的特征进行聚合。其次，设计了一种先下采样，再上采样的飞鸟目标检测网络，利用一个融合了细腻的空间信息与大感受野信息的大特征层来检测特殊的多尺度飞鸟目标。最后，简化了SimOTA动态标签分配方法，提出了SimerOTA动态标签策略，解决了因飞鸟目标不规则而导致的标签分配困难问题。论文概况图如下所示。  
<div align="center">
  <img src="https://github.com/Ziwei89/FBOD/blob/master/Illustrative_Figure/framework_github.jpeg">
</div>


本项目是该论文实现代码。本项目模型输入是连续n帧图像(以连续5帧为例)，预测飞鸟目标在中间帧的边界框信息(如果n=1, 则预测飞鸟目标在当前帧的边界框信息)

# 项目应用步骤

## 1、克隆项目到本地
```
git clone https://github.com/Ziwei89/FBOD.git
```
## 2、准备训练和测试数据

可以使用labelImg对图片进行标注，得到xml文件。  
标签文件中应包含目标边界框、目标类别以及目标的难度信息(在以后的项目中会用到目标的难度信息，在本项目中标注时不用考虑，相关代码会设置一个用不到的默认值)

### 数据组织
```  
data_root_path/  
               train/  
                     images/  
                           bird_1_000000.jpg  
                           bird_1_000001.jpg  
                           ...  
                           bird_2_000000.jpg  
                           ...  
                     labels/  
                           bird_1_000000.xml  
                           bird_1_000001.xml  
                           ...
                           bird_2_000000.xml
                           ...  
               val/  
                     images/  
                     labels/  
```  
### 然后生成数据描述txt文件用于训练和测试(训练一个txt,测试一个txt)
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
我们提供了一个脚本，可以生成这样的数据描述txt。该脚本为Data_process目录下的continuous_image_annotation.py，运行该脚本需要指定数据集路径以及模型一次推理所输入的连续图像帧数：  
```
cd Data_process
python continuous_image_annotation.py \
       --data_root_path=../dataset/FlyingBird/ \
       --input_img_num=5
cd ../ #回到项目根目录
```
运行该脚本后，将在TrainFrameword/dataloader/目录下生成两个txt文件，分别是img_label_five_continuous_difficulty_train_raw.txt和img_label_five_continuous_difficulty_val_raw.txt文件。这两个文件中的训练样本排列是顺序的，最好通过运行以下下脚本将其打乱：  
```
cd TrainFrameword/dataloader/
python shuffle_txt_lines.py \
       --input_img_num=5
cd ../../ #回到项目根目录
```
运行该脚本后，将在TrainFrameword/dataloader/目录下生成img_label_five_continuous_difficulty_train.txt和img_label_five_continuous_difficulty_val.txt两个文件。