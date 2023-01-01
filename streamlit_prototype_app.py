import streamlit as st
import cv2
import numpy as np

picture = st.camera_input("Take a picture")

if picture:
    st.image(picture)
    st.write(type(picture))
    file.write(picture.getbuffer())
