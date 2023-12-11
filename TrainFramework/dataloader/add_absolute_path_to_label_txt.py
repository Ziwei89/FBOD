
num_to_chinese_c_dic = {1:"one", 3:"three", 5:"five", 7:"seven", 9:"nine", 11:"eleven"}
continuous_img_num=5
train_absolute_path = "/home/ziwei/ziweiwork/dataset/FlyingBird/train/images/"
test_absolute_path = "/home/ziwei/ziweiwork/dataset/FlyingBird/val/images/"

if __name__ == '__main__':
    train_img_label_txt_file_all = "img_label_" + num_to_chinese_c_dic[continuous_img_num] + "_continuous_difficulty_all_raw.txt"
    list_file = open(train_img_label_txt_file_all, 'w')

    train_img_label_txt_file = "img_label_" + num_to_chinese_c_dic[continuous_img_num] + "_continuous_difficulty_train_raw.txt"
    with open(train_img_label_txt_file) as f:
        lines = f.readlines()
        for line in lines:
            line = train_absolute_path + line
            list_file.write(line)

    val_img_label_txt_file = "img_label_" + num_to_chinese_c_dic[continuous_img_num] + "_continuous_difficulty_val_raw.txt"
    with open(val_img_label_txt_file) as f:
        lines = f.readlines()
        for line in lines:
            line = test_absolute_path + line
            list_file.write(line)
    list_file.close()

    