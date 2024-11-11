from ultralytics import YOLO

# Load a model
model = YOLO("yolo11n-pose.pt")  # load an official model
# model = YOLO("path/to/best.pt")  # load a custom model

res = model.predict("pull-up.png")
# Predict with the model
results = model.track("pull-up.mp4", show=True)
# print(results[0].keypoints)     # (Batch_size=4, 17, 3)
# for result in results:
#     boxes = result.boxes  # Boxes object for bounding box outputs
#     masks = result.masks  # Masks object for segmentation masks outputs
#     keypoints = result.keypoints  # Keypoints object for pose outputs
#     probs = result.probs  # Probs object for classification outputs
#     obb = result.obb  # Oriented boxes object for OBB outputs
#     result.show()  # display to screen
#     # result.save(filename="result.jpg")  # save to disk

# View results
# for r in results:
#     print(r.keypoints)


pred = model(input)

# 画图，显示