import json
import os
import traceback
from pathlib import Path

import cv2
from train_2d_net import ResNetClassifier


# torch and lightning imports
import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder
from PIL import Image
import pandas as pd
import numpy as np

SAMPLE_RATE = 5
NUM_CLASSES = 15
# Here we define a new class to turn the ResNet model that we want to use as a feature extractor
# into a pytorch-lightning module so that we can take advantage of lightning's Trainer object.
# We aim to make it a little more general by allowing users to define the number of prediction classes.

if __name__ == "__main__":
    dataset_root = 'action_dataset'
    model_path = Path('2d', 'epoch=5-step=23436.ckpt')
    model = ResNetClassifier(num_classes=NUM_CLASSES, resnet_version=18,
                             train_path=dataset_root, test_path=None,
                             optimizer='adam', lr=1e-3,
                             batch_size=32, transfer=True, tune_fc_only=False)
    model = ResNetClassifier.load_from_checkpoint(model_path, num_classes=15, resnet_version=18, train_path=dataset_root)

    img_train = ImageFolder(Path(dataset_root, 'imgs'))
    class_to_idx = img_train.class_to_idx

    transform = transforms.Compose([
            transforms.Resize((500, 500)),
            transforms.ToTensor(),
            transforms.Normalize((0.48232,), (0.23051,))
        ])

    videos = os.listdir(Path(dataset_root, 'videos'))
    test_annotations = pd.read_csv(Path(dataset_root, 'dance-validate.csv'))
    gts = []
    preds = []
    for _, row in test_annotations.iterrows():
        try:
            if row["youtube_id"] + ".mp4" not in videos:
                continue
            gt = class_to_idx[row["label"]]
            cap = cv2.VideoCapture(str(Path(dataset_root, 'videos', row["youtube_id"] + ".mp4")))
            success, img = cap.read()
            fno = 1
            video_pred = np.zeros(NUM_CLASSES)
            while success:
                if fno % SAMPLE_RATE == 0:
                    pill_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                    transformed_img = transform(pill_img)
                    with torch.no_grad():
                        pred = model(transformed_img.unsqueeze(0))
                    video_pred += pred.numpy()[0]
                # read next frame
                success, img = cap.read()
                fno += 1
            preds.append(int(np.argmax(video_pred)))
            gts.append(gt)
        except Exception:
            traceback.print_exc()
        print(row["youtube_id"])
    test_results = (gts, preds)
    with open("test_results.json", "w") as outfile:
        json.dump(test_results, outfile)
