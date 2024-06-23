import os
import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt

# CUDA 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#print(f"Using device: {device}")

# 데이터셋 경로 설정
train_dir = 'train_data'
test_dir = 'test_data'

# 이미지 변환 설정
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

# 데이터셋 로드
train_dataset = ImageFolder(root=train_dir, transform=transform)
test_dataset = ImageFolder(root=test_dir, transform=transform)

# 데이터 로더 설정
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 클래스 이름 출력
class_names = train_dataset.classes
#print("클래스:", class_names)


# 모델 정의
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.conv3 = nn.Conv2d(64, 128, 3, 1)
        self.fc1 = nn.Linear(128 * 14 * 14, 256)
        self.fc2 = nn.Linear(256, 2)  # 출력 클래스: 여자, 남자

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2)
        x = x.view(-1, 128 * 14 * 14)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


model = SimpleCNN().to(device)

# # 손실 함수 및 옵티마이저 정의
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=0.001)
#
# # 모델 학습
# num_epochs = 25
# best_accuracy = 0.0
#
# for epoch in range(num_epochs):
#     model.train()
#     running_loss = 0.0
#     for images, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
#         images, labels = images.to(device), labels.to(device)
#         optimizer.zero_grad()
#         outputs = model(images)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()
#         running_loss += loss.item()
#
#     print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}")
#
#     # 모델 평가
#     correct = 0
#     total = 0
#     model.eval()
#     with torch.no_grad():
#         for images, labels in test_loader:
#             images, labels = images.to(device), labels.to(device)
#             outputs = model(images)
#             _, predicted = torch.max(outputs.data, 1)
#             total += labels.size(0)
#             correct += (predicted == labels).sum().item()
#
#     accuracy = 100 * correct / total
#     print(f"Test Accuracy: {accuracy:.2f}%")
#
#     # 가장 좋은 모델 저장
#     if accuracy > best_accuracy:
#         best_accuracy = accuracy
#         torch.save(model.state_dict(), 'best_model.pth')
#         print("Saved Best Model")
#
# print("Training complete.")
#
# # 학습된 모델 로드
# model.load_state_dict(torch.load('best_model.pth'))
# model.eval()

# 이미지 전처리 설정 (학습 시 사용한 설정과 동일)
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])


# 이미지 예측 함수 정의
def predict_image(image_path):
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image)
        probabilities = F.softmax(outputs, dim=1)
        _, predicted = torch.max(outputs, 1)
        predicted_class = predicted.item()
        confidence = probabilities[0][predicted_class].item()

    return predicted_class, confidence

os.environ['KMP_DUPLICATE_LIB_OK']='True'


# 새로운 이미지 예측
image_path = 'image/다운로드 (3).jpg'  # 예측할 이미지 경로
predicted_class, confidence = predict_image(image_path)

# 클래스 이름 정의 (ImageFolder 사용 시 클래스 인덱스는 자동으로 설정됨)
class_names = ['Female', 'Male']

print(f"Prediction: {class_names[predicted_class]}")
print(f"Confidence: {confidence * 100:.2f}%")

# 예측 결과를 시각화
image = Image.open(image_path).convert('RGB')
plt.imshow(image)
plt.title(f"Prediction: {class_names[predicted_class]}, Confidence: {confidence * 100:.2f}%")
plt.axis('off')
plt.show()