import streamlit as st
import joblib
import easyocr
import cv2
import numpy as np
from PIL import Image
from sklearn.feature_extraction.text import CountVectorizer

# Load model dan vectorizer
svm_model = joblib.load("svm_model.joblib")
rf_model = joblib.load("rf_model.joblib")
vectorizer = joblib.load("vectorizer.joblib")

# Inisialisasi EasyOCR
reader = easyocr.Reader(['en'])

st.title("Klasifikasi Teks & Gambar Menggunakan SVM dan Random Forest")


st.header("Input Gambar (OCR ke Teks)")
uploaded_file = st.file_uploader("Unggah gambar", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    img_array = np.array(image)

    st.image(image, caption='Gambar yang Diunggah', use_column_width=True)

    # OCR
    result = reader.readtext(img_array)
    extracted_text = " ".join([res[1] for res in result])

    st.subheader("Teks Hasil OCR:")
    st.write(extracted_text)

    if extracted_text.strip():
        text_vectorized = vectorizer.transform([extracted_text])
        svm_pred_img = svm_model.predict(text_vectorized)[0]
        rf_pred_img = rf_model.predict(text_vectorized)[0]

        st.subheader("Hasil Prediksi dari Gambar:")
        st.write(f"SVM: {svm_pred_img}")
        st.write(f"Random Forest: {rf_pred_img}")
    else:
        st.warning("Tidak ditemukan teks pada gambar.")
