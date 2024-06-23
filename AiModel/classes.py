import os
import shutil

# CelebA 데이터셋 디렉토리 경로 설정
celeba_dir = 'path_to_celeba_dataset'
images_dir = os.path.join(celeba_dir, 'img_align_celeba')  # 이미지 폴더 경로
attr_file = os.path.join(celeba_dir, 'list_attr_celeba.txt')  # 속성 파일 경로

# 여자와 남자 이미지를 저장할 폴더 경로
female_dir = 'female_images'
male_dir = 'male_images'

# 폴더 생성 (이미 생성되어 있다면 생략 가능)
os.makedirs(female_dir, exist_ok=True)
os.makedirs(male_dir, exist_ok=True)

# 속성 파일에서 성별 정보 추출
with open(attr_file, 'r') as f:
    lines = f.readlines()
    attr_header = lines[1].split()  # 속성 헤더 (첫 번째 줄은 파일 정보)

    # 성별 속성의 인덱스 찾기
    gender_idx = attr_header.index('Male') + 1  # Male 컬럼의 인덱스 (+1은 첫 번째 열이 순번이므로)

    # 각 이미지 파일에 대해 성별 정보를 확인하고 분류
    for line in lines[2:]:  # 두 번째 줄부터 속성 데이터
        tokens = line.split()
        img_filename = tokens[0]
        is_male = int(tokens[gender_idx]) == 1

        # 이미지 파일을 적절한 폴더로 복사 또는 이동
        src_path = os.path.join(images_dir, img_filename)
        if is_male:
            dst_path = os.path.join(male_dir, img_filename)
        else:
            dst_path = os.path.join(female_dir, img_filename)

        shutil.copy(src_path, dst_path)

print("이미지 분류 완료.")