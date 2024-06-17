import cv2
import os

def preprocess_and_save(image_path, output_dir, file_index):
    # 이미지 읽기
    img = cv2.imread(image_path)

    # 이미지 리사이징
    resized_img = cv2.resize(img, (448, 448))

    # HSV로 변환
    hsv_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2HSV)

    # 파일 이름 생성 및 이미지 저장
    cv2.imwrite(os.path.join(output_dir, f"{file_index}.jpg"), hsv_img)

def process_all_images(source_dir, output_dir):
    # 파일 인덱스 초기화
    file_index = 1

    # 모든 파일에 대해 반복
    for root, dirs, files in os.walk(source_dir):
        for file in files:
            # 파일이 이미지인지 확인
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(root, file)
                preprocess_and_save(image_path, output_dir, file_index)
                file_index += 1  # 파일 인덱스 증가

# 원본 이미지 폴더와 결과 이미지를 저장할 폴더
source_dir = 'Fire'
output_dir = 'Fire'

# 이미지 처리 및 저장
process_all_images(source_dir, output_dir)
