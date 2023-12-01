

cross_vx_pathes = ["cross_v1/", "cross_v2/", "cross_v3/", "cross_v4/", "cross_v5/"]
if __name__ == '__main__':

    for cross_vx_path in cross_vx_pathes:
        out = open(cross_vx_path + "test_video_name_new.txt", "w")
        lines = []
        with open(cross_vx_path + "test_video_name.txt", "r") as infile:
            for line in infile:
                lines.append(line)
        new_lines = sorted(lines, key = lambda x: int((x.strip()).split(".")[0].split("_")[-1]))
        for line in new_lines:
            out.write(line)
        out.close()