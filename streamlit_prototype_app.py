import streamlit as st
import cv2
import numpy as np

img_file_buffer = st.camera_input("Take a picture")

st.save("img_file_buffer")

if img_file_buffer is not None:
    # To read image file buffer as bytes:
    bytes_data = img_file_buffer.getvalue()
