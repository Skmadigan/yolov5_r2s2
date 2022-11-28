'''
from roboflow import Roboflow
rf = Roboflow(api_key="5YA4v3Hu51FDJqSWBS5n")
project = rf.workspace().project("taco-mqclx")
model = project.version(2).model

# infer on a local image
print(model.predict("Sarah_Litter.png", confidence=40, overlap=30).json())

# visualize your prediction
# model.predict("your_image.jpg", confidence=40, overlap=30).save("prediction.jpg")

# infer on an image hosted elsewhere
# print(model.predict("URL_OF_YOUR_IMAGE", hosted=True, confidence=40, overlap=30).json())

#!pip install roboflow
'''
from roboflow import Roboflow
rf = Roboflow(api_key="5YA4v3Hu51FDJqSWBS5n")
project = rf.workspace("divya-lzcld").project("taco-mqclx")
dataset = project.version(3).download("yolov5")