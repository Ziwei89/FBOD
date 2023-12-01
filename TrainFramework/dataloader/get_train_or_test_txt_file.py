
num_to_chinese_c_dic = {1:"one", 3:"three", 5:"five", 7:"seven", 9:"nine", 11:"eleven"}
continuous_img_num=5
cross_vx_path = "cross_v5/"

if __name__ == '__main__':
    train_img_label_txt_file = cross_vx_path + "img_label_" + num_to_chinese_c_dic[continuous_img_num] + "_continuous_difficulty_train_raw.txt"
    train_list_file = open(train_img_label_txt_file, 'w')
    test_img_label_txt_file = cross_vx_path + "img_label_" + num_to_chinese_c_dic[continuous_img_num] + "_continuous_difficulty_val_raw.txt"
    test_list_file = open(test_img_label_txt_file, 'w')

    img_label_five_continuous_difficulty_all_raw_file = "img_label_" + num_to_chinese_c_dic[continuous_img_num] + "_continuous_difficulty_all_raw.txt"

    test_video_name_file = cross_vx_path + "test_video_name.txt"
    test_video_name_list = []
    with open(test_video_name_file) as f:
        test_video_name_lines = f.readlines()
        for line in test_video_name_lines:
            test_video_name = line.strip()
            test_video_name_list.append(test_video_name)
    print(test_video_name_list)

    with open(img_label_five_continuous_difficulty_all_raw_file) as f:
        lines = f.readlines()
        for line in lines:
            image_name = line.split(" ")[0].split("/")[-1]
            video_name = image_name.split("_")[0] + "_" + image_name.split("_")[1] + ".mp4"
            if video_name in test_video_name_list:
                test_list_file.write(line)
            else:
                train_list_file.write(line)
    train_list_file.close()
    test_list_file.close()