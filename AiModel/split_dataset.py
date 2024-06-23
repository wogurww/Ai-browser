import os
import shutil
import numpy as np
from sklearn.model_selection import train_test_split

# 경로 설정
female_dir = 'female_images'
male_dir = 'male_images'
train_dir = 'train_data'
test_dir = 'test_data'

# 폴더 생성 (이미 생성되어 있다면 생략 가능)
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)
os.makedirs(os.path.join(train_dir, 'female'), exist_ok=True)
os.makedirs(os.path.join(train_dir, 'male'), exist_ok=True)
os.makedirs(os.path.join(test_dir, 'female'), exist_ok=True)
os.makedirs(os.path.join(test_dir, 'male'), exist_ok=True)


def split_and_copy_images(source_dir, train_target_dir, test_target_dir, test_size=0.2):
    # 이미지 파일 리스트 가져오기
    file_list = [f for f in os.listdir(source_dir) if os.path.isfile(os.path.join(source_dir, f))]

    # 학습 데이터와 테스트 데이터로 분할
    train_files, test_files = train_test_split(file_list, test_size=test_size, random_state=42)

    # 학습 데이터 복사
    for file in train_files:
        shutil.copy(os.path.join(source_dir, file), os.path.join(train_target_dir, file))

    # 테스트 데이터 복사
    for file in test_files:
        shutil.copy(os.path.join(source_dir, file), os.path.join(test_target_dir, file))


# 여자 이미지 분류
split_and_copy_images(female_dir, os.path.join(train_dir, 'female'), os.path.join(test_dir, 'female'))

# 남자 이미지 분류
split_and_copy_images(male_dir, os.path.join(train_dir, 'male'), os.path.join(test_dir, 'male'))

print("이미지 학습/테스트 분류 완료.")