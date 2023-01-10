import shutil
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

path_input_data = Path("data_old")
train_txt_file = "data/train.txt"
valid_txt_file = "data/val.txt"
class_name = [
    "belly_dancing",
    "breakdancing",
    "country_line_dancing",
    "dancing_ballet",
    "dancing_charleston",
    "dancing_gangnam_style",
    "dancing_macarena",
    "jumpstyle_dancing",
    "mosh_pit_dancing",
    "robot_dancing",
    "salsa_dancing",
    "square_dancing",
    "swing_dancing",
    "tango_dancing",
    "tap_dancing",
]


def data_preprarion(out_txt_file, data, typ):

    for video in tqdm(data):
        # что-то ломаное в датасете
        try:
            video_path = str(video).split("/")[-1]
            name = video_path.replace(".mp4", "")
            video_df = data_df[data_df["youtube_id"] == name]
            label = list(video_df["label"])[0]
            label = label.replace(" ", "_")
        except:
            pass

        save_dataset_path = f"data/{typ}/{label}"
        Path(save_dataset_path).mkdir(parents=True, exist_ok=True)
        save_name_path = save_dataset_path + "/" + str(video).split("/")[-1]
        shutil.copyfile((video), save_name_path)

        number_class = class_name.index(label)
        with open(out_txt_file, "a") as f:
            f.write(f"{label}/{name}.mp4" + " " + str(number_class + 1) + "\n")


if __name__ == "__main__":
    video_path_list = [x for x in path_input_data.glob("**/*.mp4")]
    data_df = pd.read_csv("dance-train.csv")
    
    train_data, valid_data = train_test_split(
        video_path_list, test_size=0.2, random_state=43)

    print(f"train: {len(train_data)} val: {len(valid_data)}")

    data_preprarion(train_txt_file, train_data, "train")
    data_preprarion(valid_txt_file, valid_data, "val")
