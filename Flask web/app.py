from flask import Flask, render_template, request
from PIL import Image
import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
from model import SimpleCNN  # assuming your model is defined in a file named model.py

app = Flask(__name__)

# CUDA 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 모델 초기화 및 가중치 로드
model = SimpleCNN().to(device)
model.load_state_dict(torch.load('best_model.pth'))
model.eval()

# 이미지 전처리 설정
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

# 클래스 이름 정의
class_names = ['Female', 'Male']

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

# 라우트 및 뷰 함수 정의
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        image_file = request.files['image']
        if image_file:
            image_path = f"static/image/{image_file.filename}"
            image_file.save(image_path)
            predicted_class, confidence = predict_image(image_path)
            return render_template('index.html', image_file=image_file.filename, class_name=class_names[predicted_class], confidence=confidence)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)