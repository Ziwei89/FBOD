# FBOD
Flying bird object detection in surveillance video

本项目模型输入是连续n帧图像(以连续5帧为例)，预测飞鸟目标在中间帧的边界框信息(如果n=1, 则预测飞鸟目标在当前帧的边界框信息)

# 项目应用步骤

## 1、克隆项目到本地

git clone https://github.com/Ziwei89/FBOD.git

## 2、准备训练和测试数据

可以使用labelImg对图片进行标注，得到xml文件。  
标签文件中应包含目标边界框、目标类别以及目标的难度信息(在以后的项目中会用到目标的难度信息，在本项目中标注是不用考虑，使用labelImg默认的难度信息即可)

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
