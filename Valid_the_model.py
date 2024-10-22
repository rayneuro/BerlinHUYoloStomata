from ultralytics import YOLO


model = YOLO('/home/ray/BerlinYoloStomata/runs/obb/BerlinHU-T22Without2/weights/best.pt')


model.predict('/home/ray/datasets/2%data/images/train' ,  save=True)