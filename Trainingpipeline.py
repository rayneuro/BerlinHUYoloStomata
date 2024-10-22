from ultralytics import YOLO


# Initialize model

model = YOLO('yolov8m-obb.pt')

# Train the model

model.train(data='trainWithoutT22.yaml', epochs=1300,name = 'BerlinHU-T22Without')

