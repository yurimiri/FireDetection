import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt

# 이미지 로드
image_path = '1.jpg'  # 이미지 경로를 입력하세요
image = Image.open(image_path)

# 각 증강 기법 정의
augmentation_transforms = {
    'Random Horizontal Flip': transforms.RandomHorizontalFlip(p=1.0),
    'Random Vertical Flip': transforms.RandomVerticalFlip(p=1.0),
    'Random Rotation 90°': transforms.RandomRotation(degrees=(90, 90)),
    'Random Color Distortion': transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
    'Random Adjust Contrast': transforms.RandomAdjustSharpness(sharpness_factor=2)
}

# 증강 결과를 저장할 리스트
augmented_images = []

# 원본 이미지 추가
augmented_images.append(('Original', image))

# 각 증강 기법을 원본 이미지에 적용
for name, transform in augmentation_transforms.items():
    augmented_image = transform(image)
    augmented_images.append((name, augmented_image))

# 증강 결과 시각화
plt.figure(figsize=(15, 10))
for i, (name, img) in enumerate(augmented_images):
    plt.subplot(2, 3, i+1)
    plt.imshow(img)
    plt.title(name)
    plt.axis('off')

plt.show()
