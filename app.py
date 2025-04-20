#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 20 13:52:15 2025

@author: chanathiptee
"""

import streamlit as st
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, decode_predictions, preprocess_input
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import pickle

with open('model.pkl', 'rb') as f :
    model = pickle.load(f)

st.title("Image Classification with MobileNetv2 by Chanathip Sirisrisermwong")

upload_file = st.file_uploader("Upload image:", type=["jpg", "jpeg", "png"])

if upload_file is not None:
    img = Image.open(upload_file)
    st.image(img, caption="Upload Image")
    
    img = img.resize((224,224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    
    preds = model.predict(x)
    top_preds = decode_predictions(preds, top=3)[0]
    
    st.subheader("Prediction:")
    for  i, preds in enumerate(top_preds):
        st.write(f"{i+1}. **{preds[1]}** — {round(preds[2]*100, 2)}%")