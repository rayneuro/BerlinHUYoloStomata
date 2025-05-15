## Introduction
Based on YOLOv8 object detection framework.

With Coordinate Attention for spatial awareness.

Efficient batch processing for large-scale image inference.



## Detection Pipeline : How to run the detection pipeline
### Step 1: 
### Puts the images folders in DetectData and enter the folder name.


```bash
/DetectData/
└── your_images_folder/  
    ├── [ex : 001.jpg,001.png] 
    ......

```


### Step 2: 
### convert the Yolov8 predict .csv file to .txt

### Converting YOLOv8 Predict OBB Format `.txt` Files to `.csv`

This guide explains how to convert YOLOv8 Oriented Bounding Box (OBB) format `.txt` files into a `.csv` file format for easier analysis or processing.

### YOLOv8 OBB `.txt` File Format

In the YOLOv8 OBB `.txt` format, each line in the file corresponds to a detected object in an image, with the following structure:

Where:
- `class_id`: The class label of the detected object.
- `x1, y1`, `x2, y2`, `x3, y3`, `x4, y4`: The coordinates of the four corners of the oriented bounding box.
- `confidence`: The confidence score of the detection.


Convert to
- `class id` : The class label of the detected object.
- `center x` , `center y` : The center of the rotated bounding box
- `w` , `h` : the width and height of rotated bounding box  
- `angle` : the angle of rotate

Corresponds to the column in the csv file:
`['class' ,'boundingbox_x' , 'boundingbox_y','boundingbox_width' ,'boundingbox_height' , 'angle' , 'confidence', 'File Name']`

The final output file is [Name of the folder].csv





