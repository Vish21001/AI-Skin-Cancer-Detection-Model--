import os
import cv2
import numpy as np
import pandas as pd
from tensorflow.keras.utils import to_categorical

DATA_DIR = "data/skin/"
IMG_SIZE = 128
ANNOTATIONS_FILE = "annotations.csv"  # columns: filename,label

def load_images():
    df = pd.read_csv(ANNOTATIONS_FILE)
    class_names = sorted(df['label'].unique())
    class_to_index = {name:i for i,name in enumerate(class_names)}
    images = []
    labels = []
    for _, row in df.iterrows():
        img_path = os.path.join(DATA_DIR, row['filename'])
        img = cv2.imread(img_path)
        if img is not None:
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            images.append(img)
            labels.append(class_to_index[row['label']])
    images = np.array(images)/255.0
    labels = to_categorical(np.array(labels), num_classes=len(class_names))
    return images, labels, class_names
