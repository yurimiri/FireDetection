import streamlit as st
from PIL import Image
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

sys.path.insert(0, './yolov5')

# 모델 예측 함수 가정(여기에 실제 모델 예측 코드를 넣으세요)
def predict_model1(image):
    model_flame = torch.hub.load('ultralytics/yolov5', 'custom',
                                 path='yoloflame2.pt')
    model_smoke = torch.hub.load('ultralytics/yolov5', 'custom',
                                 path='yolosmoke3.pt')

    model_flame.eval()
    model_smoke.eval()

    # 모델을 사용하여 예측 수행
    results_flame = model_flame(image)
    results_smoke = model_smoke(image)

    # 결과 이미지 병합
    results_combined = np.maximum(np.squeeze(results_flame.render()), np.squeeze(results_smoke.render()))
    result_image = Image.fromarray(results_combined.astype('uint8'), 'RGB')

    return result_image

def predict_model2(image):
    model = load_model('fire_detection_model.keras.keras')

    image_path = 'fire_641.jpg'

    img = image.load_img(image_path, target_size=(299, 299))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    prediction = model.predict(img_array)

    class_labels = {0: 'Flame', 1: 'Smoke', 2: 'Neutral'}

    predicted_class_index = np.argmax(prediction)
    predicted_class_label = class_labels[predicted_class_index]
    confidence = prediction[0][predicted_class_index]

    return predicted_class_label

# Streamlit 애플리케이션의 제목
st.title("Fire Detection")

# 사용자가 모델을 선택할 수 있도록 선택 상자를 생성
model_choice = st.sidebar.selectbox("Select Model", ("Yolov5", "Inception_Resnet_V2"))

# 이미지 업로드
uploaded_file = st.sidebar.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

# 업로드된 이미지가 있을 경우
if uploaded_file is not None:
    image = Image.open(uploaded_file)

    # 이미지와 결과를 나란히 표시하기 위해 컬럼 생성
    col1, col2 = st.columns(2)

    # 왼쪽 컬럼에 이미지 표시
    col1.header("Image")
    col1.image(image, use_column_width=True)

    # 예측하기 버튼
    if st.sidebar.button("Detection"):
        # 선택된 모델에 따라 예측 결과를 가져옴
        if model_choice == "Yolov5":
            result = predict_model1(image)
        else:
            result = predict_model2(image)

        # 오른쪽 컬럼에 예측 결과 표시
        col2.header("Result")
        col2.write(result)
