import os
from pathlib import Path

yolov5_path = 'dataset' # 여기에 yolov5 디렉토리의 정확한 경로를 입력하세요.
if os.path.exists(yolov5_path):
    os.chdir(yolov5_path)
else:
    print(f"디렉토리를 찾을 수 없습니다: {yolov5_path}")

# 모델 학습(smoke : 16 50 > 8 20 > 10 30 / fire : 16 50 > 8 25 > 10 30)
os.system('python train.py --img 448 --batch 10 --epochs 30 --data dataset.yaml --weights yolov5s.pt --project ../trained_model --name fire_detection')

# 모델 평가
os.system('python val.py --data dataset.yaml --weights ../trained_model/fire_detection/weights/best.pt --img 448 --batch 10 --project ../trained_model --name fire_detection_val')

# 모델 테스트
os.system('python val.py --data dataset.yaml --weights ../trained_model/fire_detection/weights/best.pt --img 448 --batch 10 --project ../trained_model --name fire_detection_test --task test')