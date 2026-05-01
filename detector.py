# Pray to the machine spririt. Thy lord may i ask for your blessin. im willing to offer my life's span for thy lord.
from ultralytics import YOLO

model = YOLO("yolo26n.pt")

# TODO: take pics, label and train model
# results = model.train(data="NOT_AVALABLE", epochs=100, imgsz=640)

trained_model = YOLO("runs/detect/train/weights/best.pt")

results = trained_model("./people.jpg")
print(results[0].names)

