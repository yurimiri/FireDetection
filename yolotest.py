import sys
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt

sys.path.insert(0, './yolov5')

# Flame detection model을 초기화하고 가중치를 로드합니다.
model_flame = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt', force_reload=True)

# Smoke detection model을 초기화하고 가중치를 로드합니다.
model_smoke = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt', force_reload=True)

# 이미지를 로드합니다.
image_path = 'smoke_5.jpg'  # 사용할 이미지의 경로로 변경하세요.
image = Image.open(image_path)

# 두 모델을 실행합니다.
results_flame = model_flame(image)
results_smoke = model_smoke(image)

# 두 결과를 병합합니다.
results_combined = np.maximum(np.squeeze(results_flame.render()), np.squeeze(results_smoke.render()))

# 병합된 결과를 시각화합니다.
plt.figure(figsize=(12, 12))
plt.imshow(results_combined)
plt.axis('off')  # 축 레이블 제거
plt.title('Combined Detection Results')

# 시각화된 이미지를 저장합니다.
plt.savefig('combined_results.jpg')  # 저장할 경로로 변경하세요.

plt.show()
