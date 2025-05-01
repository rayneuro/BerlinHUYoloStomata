## How to predict the Wheat images


### Select the images folders


```bash
/runs/
└── obb/  
    ├── [Name of the folder] 

```
You can choose the folder which contain the wheat images


## How tp convert the Yolov8 predict .csv file to .txt

### Converting YOLOv8 Predict OBB Format `.txt` Files to `.csv`

This guide explains how to convert YOLOv8 Oriented Bounding Box (OBB) format `.txt` files into a `.csv` file format for easier analysis or processing.

### YOLOv8 OBB `.txt` File Format

In the YOLOv8 OBB `.txt` format, each line in the file corresponds to a detected object in an image, with the following structure:

Where:
- `class_id`: The class label of the detected object.
- `x1, y1`, `x2, y2`, `x3, y3`, `x4, y4`: The coordinates of the four corners of the oriented bounding box.
- `confidence`: The confidence score of the detection.


Convert to
- `center x` , `center y` : The center of the rotated bounding box
- `w` , `h` : the width and height of rotated bounding box  
- `angle` : the angle of rotate

The final output file is [Name of the folder].csv


