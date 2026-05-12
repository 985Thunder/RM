from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO(r"D:\竞赛\RM\ultralytics-main\runs\detect\train\weights\best.pt")
    model.export(format='onnx')