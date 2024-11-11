import cv2 as cv
import os
import numpy as np
from ultralytics import YOLO
import torch

class AutoLabel:
    def __init__(self, path):
        self.path = path
        self.video = True
        if self.path.endswith('.jpg') or self.path.endswith('.png'):
            self.video = False
        self.output_folder = "./auto-label"
        self.frames_per_second = 1
        self.model = YOLO("yolo11n-pose.pt")
        self.size = (320, 320)
        self.visible_threshold = 0.5  # 完全可见的阈值
        self.part_visible_threshold = 0.3  # 部分可见的阈值

    def cutframe(self):
        cap = cv.VideoCapture(self.path)
        if not cap.isOpened():
            print("Error opening video file")
            return

        frame_count = 0

        while True:
            ret, frame = cap.read()

            if not ret:
                break
            frame = cv.resize(frame, (320, 320))
            if "all_frames" not in locals():
                all_frames = np.expand_dims(frame, axis=0)
            else:
                all_frames = np.concatenate([all_frames, np.expand_dims(frame, axis=0)])

            frame_count += 1
            print(frame_count)

        cap.release()
        print(f"total frames count: {frame_count}")

        return all_frames

    def label(self):
        if self.video:
            frames = self.cutframe()
        else:
            frame = cv.resize(cv.imread(self.path), self.size)
            frames = np.expand_dims(frame, axis=0)
        print(frames.shape)
        # processing
        tensor_frames = torch.from_numpy(frames).float().permute(0, 3, 1, 2) / 255.0
        print(tensor_frames.shape)
        pred = self.model.predict(tensor_frames, conf=0.25)
        # print(pred.shape)
        self.save(tensor_frames, pred)
        return pred

    def save(self, tensor, pred):

        os.makedirs(self.output_folder, exist_ok=True)
        os.makedirs(os.path.join(self.output_folder, "images"), exist_ok=True)
        os.makedirs(os.path.join(self.output_folder, "labels"), exist_ok=True)
        for index in range(tensor.shape[0]):
            image = tensor[index].permute(1, 2, 0).numpy()
            image = (image * 255).astype(np.uint8)
            cv.imwrite(os.path.join(self.output_folder, "images", f"{index}.png"), image)

            k = pred[index].keypoints
            boxes = pred[index].boxes
            conf = k.conf
            xyn = k.xyn
            info = torch.cat((xyn, conf.unsqueeze(2)), dim=2)
            with open(os.path.join(self.output_folder, "labels", f"{index}.txt"), "w") as f:
                for i in range(info.shape[0]):
                    box_str = (str(boxes.cls[i].tolist()) + " " + str(boxes.xywhn[i].tolist())).replace("[", "").replace("]", "").replace(",", " ")
                    box_str = " ".join(box_str.split())
                    f.write(box_str)
                    for j in range(info.shape[1]):
                        keypoints_str = str(info[i, j].tolist()).replace("[", "").replace("]", "").replace(",", "")
                        keypoints_str = " ".join(keypoints_str.split())
                        f.write(keypoints_str)
                    f.write("\n")





a = AutoLabel("/Users/shijunshen/Documents/Code/PycharmProjects/Posture_YOLOv11/datasets/coco8-pose/images/train/000000000077.jpg")
a.label()

