# Pray to the machine spririt. Thy lord may i ask for your blessin. im willing to offer my life's span for thy lord.
from ultralytics import YOLO

model = YOLO("yolo26n.pt")

model.train(data="datasets/orange-juice/data.yaml")

"""
trained_model = YOLO("runs/detect/train/weights/best.pt")

results = trained_model("./people.jpg")
print(results[0].names)
"""

