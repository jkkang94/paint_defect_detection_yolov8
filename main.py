import streamlit as st
from PIL import Image

# 사진 업로드 위젯
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # PIL로 이미지를 열기
    image = Image.open(uploaded_file)
    
    # 이미지 표시
    st.image(image, caption='Uploaded Image.', use_column_width=True)
