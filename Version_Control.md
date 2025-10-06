Version 2 <wheat detection model improvement with barley>
Add the barley training result in C:\Users\oscar\Desktop\BerlinHUYoloStomata\runs\obb\BerlinHU-T22Without_barley156_training.
Using C:\Users\oscar\Desktop\BerlinHUYoloStomata\runs\obb\BerlinHU-T22Without_barley156_training\weights\best.pt to do the validation.

Operation for validation (BerlinHU\_&_NTU_StomataDetect)
Set up:

1. Put the image and label in datasets/images/val & datasets/labels/val
   1.1 Like the example in the folder, you can exchange the image and label in the BerlinHUYoloStomata/datasets/images/val
   and BerlinHUYoloStomata/datasets/labels/val
2. Change the path in validBerlinHU.yaml. You can right click the datasets folder and select copy path
   2.1 path format is C:/Users/oscar/Desktop/BerlinHUYoloStomata/datasets
   2.2 please becareful the /, \ in the path format

Only need to run:

1. Change Directory
2. Run Prediction
3. Converting YOLOv8 OBB Format `.txt` Files to `.csv`
4. Validate the model
