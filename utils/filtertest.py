import cv2
import numpy as np
import matplotlib.pyplot as plt

def preprocess_and_compare(image_path):
    # 이미지 읽기
    img = cv2.imread(image_path)

    # 이미지 리사이징
    resized_img = cv2.resize(img, (448, 448))

    # RGB로 변환 (OpenCV는 기본적으로 BGR로 이미지를 읽음)
    rgb_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)

    # HSV로 변환
    hsv_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2HSV)
    # HSV 범위 설정 (예: 빨간색 범위)
    lower_bound = np.array([0, 100, 100])  # 하한값 (H, S, V)
    upper_bound = np.array([10, 255, 255])  # 상한값 (H, S, V)
    # HSV 범위에 해당하는 마스크 생성
    mask = cv2.inRange(hsv_img, lower_bound, upper_bound)
    # 원본 이미지에 마스크 적용
    result = cv2.bitwise_and(rgb_img, rgb_img, mask=mask)

    # YCrCb로 변환
    ycrcb_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2YCrCb)

    # 결과 이미지들을 화면에 표시
    plt.figure(figsize=(20, 12))
    plt.subplot(2, 2, 1), plt.imshow(rgb_img), plt.title('RGB')
    plt.subplot(2, 2, 2), plt.imshow(hsv_img), plt.title('HSV')
    plt.subplot(2, 2, 3), plt.imshow(ycrcb_img), plt.title('YCrCb')

    plt.show()

def conbination_filter(image):
    # 이미지를 HSV 색 공간으로 변환
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # 이미지를 YCrCb 색 공간으로 변환
    ycrcb_image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)

    # HSV 색 공간에서의 화재 색상 범위 설정 (예: 주황색/노란색)
    lower_hsv = np.array([0, 100, 100])
    upper_hsv = np.array([50, 255, 255])
    mask_hsv = cv2.inRange(hsv_image, lower_hsv, upper_hsv)

    # YCrCb 색 공간에서의 화재 색상 범위 설정 (예: Cr 값이 높은 영역)
    lower_ycrcb = np.array([0, 135, 85])
    upper_ycrcb = np.array([255, 180, 135])
    mask_ycrcb = cv2.inRange(ycrcb_image, lower_ycrcb, upper_ycrcb)

    # 두 마스크를 결합
    combined_mask = cv2.bitwise_and(mask_hsv, mask_ycrcb)

    # 원본 이미지에 결합된 마스크 적용
    result = cv2.bitwise_and(image, image, mask=combined_mask)

    return combined_mask, result

# 이미지 경로를 함수에 전달
preprocess_and_compare('1.jpg')

