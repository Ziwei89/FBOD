import random
num_to_chinese_c_dic = {1:"one", 3:"three", 5:"five", 7:"seven", 9:"nine", 11:"eleven"}
continuous_img_num=5
cross_vx_pathes = ["cross_v1/", "cross_v2/", "cross_v3/", "cross_v4/", "cross_v5/"]
if __name__ == '__main__':

    for cross_vx_path in cross_vx_pathes:
        out = open(cross_vx_path + "img_label_" + num_to_chinese_c_dic[continuous_img_num] + "_continuous_difficulty_train.txt", "w")
        lines = []
        with open(cross_vx_path + "img_label_" + num_to_chinese_c_dic[continuous_img_num] + "_continuous_difficulty_train_raw.txt", "r") as infile:
            for line in infile:
                lines.append(line)
            random.shuffle(lines)
            random.shuffle(lines)
            random.shuffle(lines)
            random.shuffle(lines)
        for line in lines:
            out.write(line)
        out.close()

        test_out = open(cross_vx_path + "img_label_" + num_to_chinese_c_dic[continuous_img_num] + "_continuous_difficulty_val.txt", "w")
        lines = []
        with open(cross_vx_path + "img_label_" + num_to_chinese_c_dic[continuous_img_num] + "_continuous_difficulty_val_raw.txt", "r") as infile:
            for line in infile:
                lines.append(line)
            random.shuffle(lines)
            random.shuffle(lines)
            random.shuffle(lines)
            random.shuffle(lines)
        for line in lines:
            test_out.write(line)
        test_out.close()