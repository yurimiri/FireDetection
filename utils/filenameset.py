import os

def rename_and_change_extension(folder_path, new_extension='.jpg'):
    # 폴더 내 파일 목록 가져오기
    files = os.listdir(folder_path)

    # 파일 목록 정렬 (필요에 따라 정렬 방식 변경 가능)
    files.sort()

    # 임시 이름으로 모든 파일 이름 변경
    temp_files = []
    for i, filename in enumerate(files, start=1):
        old_file_path = os.path.join(folder_path, filename)
        temp_file_path = os.path.join(folder_path, f"a_{i}")
        os.rename(old_file_path, temp_file_path)
        temp_files.append(temp_file_path)

    # 임시 이름을 가진 파일들을 순서대로 순회하며 새 이름과 확장자로 변경
    for i, temp_file_path in enumerate(temp_files, start=1):
        new_filename = f"smoke_{i}{new_extension}"
        new_file_path = os.path.join(folder_path, new_filename)
        os.rename(temp_file_path, new_file_path)
        print(f"Renamed: {temp_file_path} -> {new_file_path}")

# 사용 예시
folder_path = 'smoke'  # 폴더 경로를 여기에 입력하세요
rename_and_change_extension(folder_path)
