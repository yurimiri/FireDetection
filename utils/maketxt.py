import os

# 경로 설정
image_dir = 'Dataset/nofire/val' # 이미지 파일 디렉토리 경로
new_image_dir = 'images/val/nofire'  # 새로운 이미지 파일 디렉토리 경로
new_label_dir = 'images/val/nofire'  # 새로운 라벨 파일 디렉토리 경로

# 디렉토리 내 모든 파일 목록 가져오기
image_files = [f for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f))]

count = 1
for image_file in image_files:
    name, ext = os.path.splitext(image_file)
    new_name = f"no_fire_{count}"

    new_image_file = new_name + ext  # 새로운 이미지 파일 이름
    new_label_file = new_name + '.txt'  # 새로운 라벨 파일 이름

    # 이미지 파일 이름 변경 및 이동
    os.rename(os.path.join(image_dir, image_file), os.path.join(new_image_dir, new_image_file))

    # 빈 라벨 파일 생성
    with open(os.path.join(new_label_dir, new_label_file), 'w') as f:
        pass  # 빈 파일 생성

    count += 1

print("이미지 파일 이름 변경 및 빈 라벨 파일 생성이 완료되었습니다.")
