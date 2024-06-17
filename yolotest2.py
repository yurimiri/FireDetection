import sys
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

sys.path.insert(0, './yolov5')

smoke_model = torch.hub.load('ultralytics/yolov5', 'custom', path='models/yolosmoke.pt')
# flame_model = torch.hub.load('ultralytics/yolov5', 'custom', path='models/yoloflame.pt')

smoke_model.eval()
# flame_model.eval()

image_path = 'smoke_8.jpg'  # 사용할 이미지의 경로로 변경하세요.
image = Image.open(image_path)

smoke_result = smoke_model(image)
# flame_result = flame_model(image)

smoke_result.show()
# flame_result.show()

results_img = np.squeeze(smoke_result.render())  # 결과 이미지를 numpy 배열로 변환
plt.figure(figsize=(12, 12))
plt.imshow(results_img)
plt.axis('off')  # 축 레이블 제거
plt.show()

# results_img = np.squeeze(flame_result.render())  # 결과 이미지를 numpy 배열로 변환
# plt.figure(figsize=(12, 12))
# plt.imshow(results_img)
# plt.axis('off')  # 축 레이블 제거
# plt.show()