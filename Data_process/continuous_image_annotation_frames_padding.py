import os
import xml.etree.ElementTree as ET
import argparse

num_to_chinese_c_dic = {1:"one", 3:"three", 5:"five", 7:"seven", 9:"nine", 11:"eleven"}
def difficulty_class_to_conf(difficulty_class_level):
    return (0.875 - difficulty_class_level/4)
classes=['bird']
def convert_annotation(annotation_file, list_file):
    in_file = open(annotation_file, encoding='utf-8')
    tree=ET.parse(in_file)
    root = tree.getroot()

    if root.find('object')!=None:
        for obj in root.iter('object'):
            difficult = 0 
            if obj.find('difficult')!=None:
                difficult = obj.find('difficult').text
            obj_conf = difficulty_class_to_conf(int(difficult))
                
            cls = obj.find('name').text
            if cls not in classes:
                continue
            cls_id = classes.index(cls)
            xmlbox = obj.find('bndbox')
            b = (int(xmlbox.find('xmin').text), int(xmlbox.find('ymin').text), int(xmlbox.find('xmax').text), int(xmlbox.find('ymax').text))
            list_file.write(" " + ",".join([str(a) for a in b]) + ',' + str(cls_id) + ',' + str(obj_conf))
    else:
        list_file.write(" " + "None")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root_path', default="../dataset/FBD-SV-2024/", type=str,
                        help='data_root_path: The path of the dataset.')
    parser.add_argument('--input_img_num', default=5, type=int,
                        help='input_img_num: The continous video frames, input to the model')
    parser.add_argument('--img_ext', default=".jpg", type=str,
                        help='img_ext: The extension name of the image')
    args = parser.parse_args()

    train_img_label_txt_file = "../TrainFramework/dataloader/img_label_" + num_to_chinese_c_dic[args.input_img_num] + "_continuous_difficulty_train_raw.txt"
    val_img_label_txt_file = "../TrainFramework/dataloader/img_label_" + num_to_chinese_c_dic[args.input_img_num] + "_continuous_difficulty_val_raw.txt"

    list_file_train = open(train_img_label_txt_file, 'w')
    list_file_val = open(val_img_label_txt_file, 'w')
    list_files = [list_file_train, list_file_val]

    train_label_path = args.data_root_path + "labels/train/"
    val_label_path = args.data_root_path + "labels/val/"
    label_pathes = [train_label_path, val_label_path]

    train_image_path = args.data_root_path + "images/train/"
    val_image_path = args.data_root_path + "images/val/"
    image_pathes = [train_image_path, val_image_path]

    for list_file, label_path, image_path in zip(list_files, label_pathes, image_pathes):
        label_files = os.listdir(label_path)
        for label_file in label_files:
            if label_file.split(".")[1]!="xml":
                continue
            num = int((label_file.split(".")[0]).split("_")[-1])
            num_str = "%06d" % int(num)
            prefix_name = label_file.split(num_str)[0]

            num = num-int(args.input_img_num/2)
            Is_needed_continuous_img_exit = True
            if num < 0: #### The previous image does not exist and needs padding, the image with a sequence number greater than zero must exist.
                for i in range(0, num+args.input_img_num): ### The 0^th,1^st,..args.input_img_num^th image is needed.
                    i_str = "%06d" % int(i)
                    image_name = prefix_name + i_str + args.img_ext
                    if not os.path.exists(image_path + image_name):
                        Is_needed_continuous_img_exit = False
                        break
            else:   #### If the image at the back doesn't exist and needs padding, the previous image must exist.
                for i in range(num, num+int(args.input_img_num/2)): ### The num^th,(num+1)^st,..(num+args.input_img_num/2)^th image is needed.
                    i_str = "%06d" % int(i)
                    image_name = prefix_name + i_str + args.img_ext
                    if not os.path.exists(image_path + image_name):
                        Is_needed_continuous_img_exit = False
                        break          
            if Is_needed_continuous_img_exit: 
                num_str = "%06d" % int(num)  ## Don't consider wether the num is less than 0
                image_name = prefix_name + num_str + args.img_ext
                print(image_name)
                list_file.write(image_name)
                lable_str = convert_annotation(label_path + label_file, list_file)
                list_file.write("\n")
        list_file.close()