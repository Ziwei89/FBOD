# FBOD
Flying bird object detection in surveillance video

本项目模型输入是连续n帧图像(以连续5帧为例)，预测飞鸟目标在中间帧的边界框信息(如果n=1, 则预测飞鸟目标在当前帧的边界框信息)

# 项目应用步骤

## 1、克隆项目到本地

git clone https://github.com/Ziwei89/FBOD.git

## 2、准备训练和测试数据

可以使用labelImg对图片进行标注，得到xml文件。

### 数据组织

data_root_path/
               train/
                     images/
                           bird_1_000000.jpg
                           bird_1_000001.jpg
                           ...
                     labels/
                           bird_1_000000.xml
                           bird_1_000001.xml
                           ...
               val/
                   images/
                   labels/

### 然后生成数据描述txt文件用于训练和测试(训练一个txt,测试一个txt)
