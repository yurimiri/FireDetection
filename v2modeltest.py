import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import os

# 저장된 모델 경로
model_path = 'inception_resnet_v2_model2.h5'

class CustomScaleLayer(tf.keras.layers.Layer):
    def __init__(self, scale, **kwargs):
        super(CustomScaleLayer, self).__init__(**kwargs)
        self.scale = scale  # scale 값을 내부 속성으로 저장

    def build(self, input_shape):
        # 필요한 경우 여기에서 레이어의 가중치를 정의할 수 있습니다.
        super().build(input_shape)

    def call(self, inputs, **kwargs):
        # inputs가 리스트인 경우, 리스트의 각 요소에 대해 self.scale을 곱함
        if isinstance(inputs, list):
            return [x * self.scale for x in inputs]
        else:
            # inputs가 리스트가 아닌 경우, 직접 곱셈 수행
            return inputs * self.scale

# 모델 로딩 시에 CustomScaleLayer를 custom_objects에 등록합니다.
model = tf.keras.models.load_model(model_path, custom_objects={'CustomScaleLayer': CustomScaleLayer})

# 테스트 이미지 경로
test_image_path = 'fire_64.jpg' # 예시 경로, 실제 경로로 변경해주세요.

# 이미지 로드 및 전처리
img = image.load_img(test_image_path, target_size=(299, 299))
img_array = image.img_to_array(img)
img_array_expanded_dims = np.expand_dims(img_array, axis=0)
preprocessed_img = preprocess_input(img_array_expanded_dims)

# 예측 수행
predictions = model.predict(preprocessed_img)

print(predictions)

# 예시로, 'detections'는 모델이 이미지에서 탐지한 객체의 정보를 담고 있다고 가정합니다.
# 각 탐지된 객체는 바운딩 박스 정보(x, y, width, height)와 레이블을 포함합니다.
# detections = [
#     {'box': [100, 200, 50, 100], 'label': 'fire'},
#     {'box': [300, 400, 100, 150], 'label': 'smoke'}
# ]
#
# # 이미지 로드
# img = Image.open(test_image_path)
#
# # matplotlib로 이미지 표시
# fig, ax = plt.subplots()
# ax.imshow(img)
#
# # 탐지된 각 객체에 대한 바운딩 박스와 레이블 그리기
# for detection in detections:
#     box = detection['box']
#     rect = patches.Rectangle((box[0], box[1]), box[2], box[3], linewidth=2, edgecolor='r', facecolor='none')
#     ax.add_patch(rect)
#     plt.text(box[0], box[1]-10, detection['label'], color='red', fontsize=15)
#
# plt.axis('off')
# plt.show()
