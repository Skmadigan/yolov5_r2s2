import torch

# Model
#model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # or yolov5n - yolov5x6, custom
##model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # or yolov5n - yolov5x6, custom
#model = torch.load(r'C:\Users\skmad\OneDrive - The Ohio State University\_Senior(2022-2023)\AU22_OneDrive\ENGR_5901-01_OneDrive\Cloned_Git_Repo\YOLO_v5\yolov5\runs\train\exp8\weights\best.pt')
#print('Loaded Model I think')
#model = torch.load(r'C:\Users\skmad\OneDrive - The Ohio State University\_Senior(2022-2023)\AU22_OneDrive\ENGR_5901-01_OneDrive\Cloned_Git_Repo\YOLO_v5\yolov5\runs\train\exp8\weights\best.pt')
weights_path = r'C:\Users\skmad\Documents\r2s2\r2s2\yolo_v5_Neural_Network\yolov5\runs\train\Google_GPU_YOLOv5m_11-21-22\weights\best.pt'
#model = torch.hub.load('ultralytics/yolov5', 'custom', path_or_model=)
img = 'https://thumbs.dreamstime.com/b/pop-tab-silver-color-small-drink-can-kept-wooden-table-144049269.jpg'


#model = torch.hub.load(model, 'yolov5s', repo_or_dir = 'ultralytics/yolov5', weights=weights_path)

#model = torch.hub.load('ultralytics/yolov5', 'yolov5s', 'custom', weights_path, force_reload=True, device='cpu')  # yolov5n - yolov5x6 official model
#                                            'custom', 'path/to/best.pt')  # custom model

model = torch.hub.load('ultralytics/yolov5','custom', path = r'C:\Users\skmad\Documents\r2s2\r2s2\yolo_v5_Neural_Network\yolov5\runs\train\Google_GPU_YOLOv5m_11-21-22\weights\best.pt', force_reload = True)
#model.load_state_dict(torch.load(weights_path)['model'].state_dict())

#model = model.fuse().autoshape()


# Inference
results = model(img)#, force_reload=True)

# Results
results.show()  # or .show(), .save(), .crop(), .pandas(), etc.
print(results.xyxy[0])  # print img1 predictions (pixels)
#                   x1           y1           x2           y2   confidence        class
# tensor([[7.50637e+02, 4.37279e+01, 1.15887e+03, 7.08682e+02, 8.18137e-01, 0.00000e+00],
#         [9.33597e+01, 2.07387e+02, 1.04737e+03, 7.10224e+02, 5.78011e-01, 0.00000e+00],
#         [4.24503e+02, 4.29092e+02, 5.16300e+02, 7.16425e+02, 5.68713e-01, 2.70000e+01]])