import os

image_dir = 'images/val'  # 이미지 파일이 위치한 디렉토리 경로
label_dir = 'labels/val'  # 라벨 파일이 위치한 디렉토리 경로
new_image_dir = 'images/val'  # 변경된 이미지 파일을 저장할 디렉토리 경로
new_label_dir = 'labels/val'  # 변경된 라벨 파일을 저장할 디렉토리 경로

image_files = [f for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f))]
label_files = [f for f in os.listdir(label_dir) if os.path.isfile(os.path.join(label_dir, f))]

count = 1
for image_file in image_files:
    name, ext = os.path.splitext(image_file)
    corresponding_label_file = name + '.txt'

    if corresponding_label_file in label_files:
        new_name = f"fire_{count}"
        new_image_file = new_name + ext  # 새 이미지 파일 이름
        new_label_file = new_name + '.txt'  # 새 라벨 파일 이름

        # 이미지 파일 이름 변경 및 이동
        os.rename(os.path.join(image_dir, image_file), os.path.join(new_image_dir, new_image_file))
        # 라벨 파일 이름 변경 및 이동
        os.rename(os.path.join(label_dir, corresponding_label_file), os.path.join(new_label_dir, new_label_file))

        count += 1

print("파일 이름 변경 및 이동이 완료되었습니다.")
