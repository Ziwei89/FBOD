import random
if __name__ == '__main__':
    out = open("./img_label_five_continuous_difficulty_train.txt", "w")
    lines = []
    with open("./img_label_five_continuous_difficulty_train_raw.txt", "r") as infile:
        for line in infile:
            lines.append(line)
        random.shuffle(lines)
        random.shuffle(lines)
        random.shuffle(lines)
        random.shuffle(lines)
    for line in lines:
        out.write(line)