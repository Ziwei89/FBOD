import random
import argparse

num_to_chinese_c_dic = {1:"one", 3:"three", 5:"five", 7:"seven", 9:"nine", 11:"eleven"}
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_img_num', default=5, type=int,
                        help='input_img_num: The continous video frames, input to the model')
    args = parser.parse_args()

    train_img_label_txt_file_raw = "img_label_" + num_to_chinese_c_dic[args.input_img_num] + "_continuous_difficulty_train_raw.txt"
    val_img_label_txt_file_raw = "img_label_" + num_to_chinese_c_dic[args.input_img_num] + "_continuous_difficulty_val_raw.txt"
    in_files = [train_img_label_txt_file_raw, val_img_label_txt_file_raw]

    train_img_label_txt_file = "img_label_" + num_to_chinese_c_dic[args.input_img_num] + "_continuous_difficulty_train.txt"
    val_img_label_txt_file = "img_label_" + num_to_chinese_c_dic[args.input_img_num] + "_continuous_difficulty_val.txt"
    out_files = [train_img_label_txt_file, val_img_label_txt_file]

    for in_file, out_file in zip(in_files, out_files):
        out = open(out_file, "w")
        lines = []
        with open(in_file, "r") as infile:
            for line in infile:
                lines.append(line)
            random.shuffle(lines)
            random.shuffle(lines)
            random.shuffle(lines)
            random.shuffle(lines)
        for line in lines:
            out.write(line)
        out.close()