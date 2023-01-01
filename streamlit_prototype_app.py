# 모듈 불러오기
import streamlit as st
from streamlit_folium import st_folium
import streamlit.components.v1 as components

import folium
import cv2

import pandas as pd
import numpy as np
from datetime import datetime

import time

from keras.models import load_model
from PIL import Image, ImageOps

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

model = load_model('keras_model.h5', compile=False)

class_names = open('labels.txt', 'r', encoding='utf-8').readlines()



# 흡연구역 불러오기
smoking_area = pd.read_csv('흡연부스중구름방.csv')

# 시작좌표 및 타일지정
center = [37.5664750, 126.981846] # 하나카드좌표
tiles = ['cartodbpositron', 'Stamen Toner', 'OpenStreetMap']

st.subheader('흡연부스 지도')

# 지도에 표시
m = folium.Map(
    location = [center[0], center[1]],
    zoom_start = 18,
#     tiles = tiles[0]
)


# 중구 흡연구역 마커
for i in range(smoking_area.shape[0]):
        folium.Marker(
            location = [smoking_area.loc[i, '_Y'], smoking_area.loc[i, '_X']],
            popup = smoking_area.loc[i, 'field1'],
            tooltip = smoking_area.loc[i, 'field1'],
            icon = folium.Icon('red', icon = 'star')
        ).add_to(m)

# 보여주기
st_data = st_folium(m, height=500, width=1200)




# 실시간 CCTV // 라이브 웹캠
st.subheader("실시간 CCTV 영상")
run = st.checkbox('Run')


while run:

    img_file_buffer = st.camera_input(label='CCTV', key='CCTV')
    
    # 이미지 캡쳐
    if img_file_buffer is not None:
        bytes_data = img_file_buffer.getvalue()
        cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
        
        np.set_printoptions(suppress=True)

        data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
       
        image = Image.open(img_file_buffer)

        img_array = np.array(image)
        size = (224, 224)
        image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
        image_array = np.asarray(image)
        normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1# Load the image into the array
        data[0] = normalized_image_array

        prediction = model.predict(data)
        index = np.argmax(prediction)
        class_name = class_names[index]
        confidence_score = prediction[0][1]
        
        clean_status = round(100 - confidence_score * 100, 2)
        
        if clean_status <= 50:
            st.warning(f'청결도: {class_name[2:]}, 청결도 : {clean_status}%', icon="⚠️")
        
        
        

st.subheader('알림 기능 예시')
alert_run = st.checkbox('Alert')
trash_can = 0

while alert_run:

    time.sleep(np.random.randint(0,2))

    if trash_can >= 85:
        st.warning('쓰레기통을 비워주세요..', icon="⚠️")
    # elif (trash_can >= 0) & (trash_can < 85):
    #     st.error('쓰레기통이 비었습니다.')
        break
        
    trash_can += np.random.randint(1,6)
        
    


