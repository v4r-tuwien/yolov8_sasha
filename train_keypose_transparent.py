from ultralytics import YOLO

# Load a model
model = YOLO('yolov10s.pt')  # load a pretrained model (recommended for training)
#model = YOLO('runs/segment/train/weights/last.pt')  # load a pretrained model (recommended for training)

model.info()

# Train the model
results = model.train(data='keypose_transparent.yaml', epochs=30, imgsz=640, augment=True)

# Validate the model
metrics = model.val()  # no arguments needed, dataset and settings remembered
metrics.box.map    # map50-95(B)
metrics.box.map50  # map50(B)
metrics.box.map75  # map75(B)
metrics.box.maps   # a list contains map50-95(B) of each category