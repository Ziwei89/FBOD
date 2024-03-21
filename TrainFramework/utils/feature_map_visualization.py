import cv2
import numpy as np
import copy

def create_grayscale_heatmap(image):
    heatmap = cv2.applyColorMap(image, cv2.COLORMAP_JET)
    return heatmap

def show_feature_map_heatmap(in_feature_map, img_name="feature_map.jpg", model_input_size=(672,384), input_img_size=(1280,720), pic_save_dir="./test_output/"):
    resize_ratio = min(1.0 * model_input_size[0] / input_img_size[0], 1.0 * input_img_size[1] / input_img_size[1])
    resize_w = int(resize_ratio * input_img_size[0])
    resize_h = int(resize_ratio * input_img_size[1])

    dh = model_input_size[1] - resize_h
    top = int(dh/2)
    bottom = dh - top

    dw = model_input_size[0] - resize_w
    left = int(dw/2)
    right = dw - left

    crop_box = [left, top, model_input_size[0]-right, model_input_size[1]-bottom] #x1,y1, x2,y2

    feature_map_data_list = [in_feature_map[0, i].cpu().detach().numpy() for i in range(in_feature_map.shape[1])]
    feature_map = np.zeros(shape=(input_img_size[1], input_img_size[0]))
    for feature_map_data in feature_map_data_list:
        feature_map_data = cv2.resize(feature_map_data, model_input_size)
        feature_map_data_crop = copy.deepcopy(feature_map_data[crop_box[1]:crop_box[3], crop_box[0]:crop_box[2]])
        feature_map_data = cv2.resize(feature_map_data_crop, input_img_size)
        feature_map += feature_map_data
    pmin = np.min(feature_map)
    pmax = np.max(feature_map)
    feature_map = (feature_map - pmin) / (pmax - pmin + 0.000001) * 255
    feature_map = np.asarray(feature_map, dtype=np.uint8)
    heatmap = create_grayscale_heatmap(feature_map)
    # prefix_name = img_name.split(".")[0]
    # prefix_name_list = prefix_name.split("_")
    # new_num_str = "%06d" % (int(prefix_name_list[2]) + 2)
    # img_name = prefix_name_list[0] + "_" + prefix_name_list[1] + "_" + new_num_str + ".jpg"
    cv2.imwrite(pic_save_dir + img_name, heatmap)