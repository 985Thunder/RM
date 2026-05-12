import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from ultralytics import YOLO
if __name__ == '__main__':
    model = YOLO(r"yolo26n.pt")
    model.train(
        data=r"RM_.yaml",
        epochs=30,
        imgsz=640,
        batch=-1,
        cache='ram',
        workers=1,
    )