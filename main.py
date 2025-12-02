from ultralytics import YOLO

model = YOLO("yolov8n-seg.pt")   # pre-trained base model
model.train(
    data="dataset/data.yaml",
    epochs=80, augment=True
)
# "yolo train data=crop_data.yaml model=yolov8n.pt epochs=10 imgsz=640"
# "yolo detect train model=path/to/best.pt data=path/to/new_data.yaml epochs=80 imgsz=640"