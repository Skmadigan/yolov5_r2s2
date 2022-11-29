# SARAH MADIGAN 11/25/22
# Script written to give the location of a bounding box from the YOLOv5s model
# Model isn't trained to be super accurate, but this is a starting point for Brennan/Jonathon to use with integration
import torch

#Update this weights path for the best.py file associated with the model training you're trying to use
weights_path = r'C:\Users\skmad\Documents\r2s2\r2s2\yolo_v5_Neural_Network\yolov5\runs\train\Google_GPU_YOLOv5m_11-21-22\weights\best.pt'

#These img variables can be from online or your computer.  Replace the filepaths/URLs accordingly
img = 'https://www.photocase.com/photos/3334729-filter-cigarette-lies-on-a-wooden-table-addiction-photocase-stock-photo-large.jpeg'
img2 = r'C:\Users\skmad\Documents\r2s2\r2s2\yolo_v5_Neural_Network\yolov5\TACO-3\test\images\000005_JPG_jpg.rf.6eb81505cad603f0dd0e500f5f75d52e.jpg'
img_can_CDR = r'C:\Users\skmad\Documents\r2s2\r2s2\yolo_v5_Neural_Network\datasets\taco\test\images\000024_jpg.rf.df34892fd00bc4276c0dbe0ad8e16247.jpg'
#Loads the model using custom weights
model = torch.hub.load('ultralytics/yolov5','custom', path = weights_path, force_reload = True)

#Runs the model on whichever image you want
results = model(img_can_CDR)

# Results

#This line causes an image with the bounding box, label, and confidence level to show up
results.show()  # or .show(), .save(), .crop(), .pandas(), etc.

#Accessing information about each bounding box and displaying it 
boxes = results.pandas().xyxy[0]
print(boxes)
print("X MIN: \n", boxes.xmin)
print("X MAX: \n", boxes.xmax)
print("Y MIN: \n", boxes.ymin)
print("Y MAX: \n", boxes.ymax)
xmaxtest = boxes.xmin
print(xmaxtest)
