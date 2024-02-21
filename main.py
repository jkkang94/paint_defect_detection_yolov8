import streamlit as st
from PIL import Image
import torch
from torchvision import transforms
import cv2
import numpy as np
import requests
from pathlib import Path

# Streamlit 페이지 구성
st.title('YOLO Object Detection')
st.write('This is a simple object detection app using YOLO.')

# 사진 업로드 위젯
uploaded_file = st.sidebar.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# 설정 옵션 - 예를 들어, 감지 정확도 임곗값 설정
confidence_threshold = st.sidebar.slider("Confidence threshold", 0.0, 1.0, 0.5, 0.01)

def download_model(url, destination):
    """
    GitHub URL에서 모델 파일을 다운로드하고 지정된 위치에 저장합니다.
    """
    response = requests.get(url)
    response.raise_for_status()  # 에러 발생 시 예외 처리
    with open(destination, "wb") as f:
        f.write(response.content)
    print(f"Model downloaded to {destination}")

# 모델 URL (GitHub에서 직접 다운로드 가능한 raw 파일 링크)
model_url = "https://github.com/yourusername/yourrepository/raw/main/path/to/your/model.pt"

# 모델 저장 위치
model_path = "path/to/save/model.pt"

# 파일이 이미 존재하지 않는 경우에만 다운로드
if not Path(model_path).is_file():
    download_model(model_url, model_path)
else:
    print("Model already downloaded.")
    
# 이미지 처리 및 객체 감지 함수
def detect_objects(image):
    # 이미지를 모델 입력에 맞게 변환
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    input_image = transform(image).unsqueeze(0)
    
    # 모델 실행
    model.eval()
    with torch.no_grad():
        detections = model(input_image)
    
    # 결과 처리
    results = detections.xyxy[0].numpy()  # 감지된 객체의 바운딩 박스
    return results

# "Detect" 버튼
if st.button('Detect'):
    if uploaded_file is not None:
        # 이미지를 PIL로 로드
        image = Image.open(uploaded_file).convert('RGB')
        image_np = np.array(image)

        # 객체 감지 실행
        results = detect_objects(image)
        
        # 감지 결과를 이미지에 표시
        for result in results:
            if result[4] >= confidence_threshold:
                cv2.rectangle(image_np, (int(result[0]), int(result[1])), (int(result[2]), int(result[3])), (255, 0, 0), 2)
                cv2.putText(image_np, f'{model.names[int(result[5])]} {result[4]:.2f}', (int(result[0]), int(result[1]-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,0,0), 2)
        
        # 변환된 이미지 표시
        st.image(image_np, caption='Detected Objects', use_column_width=True)
    else:
        st.write("No file uploaded.")
