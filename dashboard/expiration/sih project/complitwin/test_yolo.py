from ultralytics import YOLO
model = YOLO("yolo26n.pt")
results=model("product.jpg")
for result in results:
    result.show()