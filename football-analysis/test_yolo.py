from ultralytics import YOLO

print("Loading YOLO26n...")

model = YOLO("yolo26n.pt")

print("YOLO26n loaded successfully!")
print("Classes:")
print(model.names)