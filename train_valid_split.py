from sklearn.model_selection import train_test_split
import os 
import shutil

# Get all the images name from the directory
images = os.listdir('/home/ray/datasets/AllDataFiveClass(WithoutT22L600)/images')


for name in images:
    if name == 'train' or name == 'valid':
        images.remove(name)

names = [image.split('.')[0] for image in images]

#imagenames = [ 'home/ray/datasets/AllDataFiveClass(without T22L600)/images'+name+ for name in names] /

# Split the data into train and validation set
train, valid = train_test_split(names, test_size=0.1, random_state=40) 

# Save the train and validation set into a file
for name in train:
    shutil.copy('/home/ray/datasets/AllDataFiveClass(WithoutT22L600)/images/'+name+'.jpg', '/home/ray/datasets/AllDataFiveClass(WithoutT22L600)/images/train')
    shutil.copy('/home/ray/datasets/AllDataFiveClass(WithoutT22L600)/labels/'+name+'.txt', '/home/ray/datasets/AllDataFiveClass(WithoutT22L600)/labels/train')
    
for name in valid:
    shutil.copy('/home/ray/datasets/AllDataFiveClass(WithoutT22L600)/images/'+name+'.jpg', '/home/ray/datasets/AllDataFiveClass(WithoutT22L600)/images/valid')
    shutil.copy('/home/ray/datasets/AllDataFiveClass(WithoutT22L600)/labels/'+name+'.txt', '/home/ray/datasets/AllDataFiveClass(WithoutT22L600)/labels/valid')
    