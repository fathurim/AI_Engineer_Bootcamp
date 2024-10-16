import pickle
import streamlit as st
from streamlit_option_menu import option_menu
import joblib
import numpy as np
from huggingface_hub import hf_hub_download
import torch
from torchvision import models, transforms
from PIL import Image

# Mengunduh model dari Hugging Face Hub
diabetes_model_path = hf_hub_download(repo_id="fathurim/diabetes_prediction", filename="diabetes_prediction.pkl")
heart_disease_model_path = hf_hub_download(repo_id="fathurim/heart_disease_prediction", filename="heart_disease.pkl")
pneumonia_model_path = hf_hub_download(repo_id="fathurim/pneumonia_resnet", filename="pneumonia_resnet18.pt")

# Memuat model
diabetes_model = joblib.load(diabetes_model_path)
heart_disease_model = joblib.load(heart_disease_model_path)

# Memuat model pneumonia
pneumonia_model = models.resnet18(num_classes=2)
pneumonia_model.load_state_dict(torch.load(pneumonia_model_path, map_location=torch.device('cpu')))
pneumonia_model.eval()

# Transformasi untuk input gambar
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Normalize sesuai ImageNet
])

# Kelas label untuk prediksi pneumonia
class_names = ['NORMAL', 'PNEUMONIA']

# Fungsi untuk membuat prediksi pneumonia
def predict(image):
    # Preprocess image
    image = transform(image).unsqueeze(0)  # Tambahkan dimensi batch
    pneumonia_model.eval()
    with torch.no_grad():
        output = pneumonia_model(image)
        _, predicted = torch.max(output, 1)
        label = class_names[predicted.item()]

    return label

# Set page configuration
st.set_page_config(page_title="Health Assistant", layout="wide", page_icon="🧑‍⚕️")

# Sidebar for navigation
with st.sidebar:
    selected = option_menu('Multiple Disease Prediction System',
                           ['Diabetes Prediction', 'Heart Disease Prediction', 'Pneumonia Prediction'],
                           menu_icon='hospital-fill',
                           icons=['activity', 'heart', 'person'],
                           default_index=0)

# Diabetes Prediction Page
if selected == 'Diabetes Prediction':
    st.title('Diabetes Prediction using ML')
    col1, col2, col3 = st.columns(3)

    with col1:
        Pregnancies = st.text_input('Number of Pregnancies')
    with col2:
        Glucose = st.text_input('Glucose Level')
    with col3:
        BloodPressure = st.text_input('Blood Pressure value')
    with col1:
        SkinThickness = st.text_input('Skin Thickness value')
    with col2:
        Insulin = st.text_input('Insulin Level')
    with col3:
        BMI = st.text_input('BMI value')
    with col1:
        DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function value')
    with col2:
        Age = st.text_input('Age of the Person')

    if st.button('Diabetes Test Result'):
        user_input = [Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]
        user_input = [float(x) for x in user_input]
        diab_prediction = diabetes_model.predict([user_input])

        if diab_prediction[0] == 1:
            st.success('The person is diabetic')
        else:
            st.success('The person is not diabetic')

# Heart Disease Prediction Page
if selected == 'Heart Disease Prediction':
    st.title('Heart Disease Prediction using ML')
    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.text_input('Age')
    with col2:
        sex = st.text_input('Sex')
    with col3:
        cp = st.text_input('Chest Pain types')
    with col1:
        trestbps = st.text_input('Resting Blood Pressure')
    with col2:
        chol = st.text_input('Serum Cholestoral in mg/dl')
    with col3:
        fbs = st.text_input('Fasting Blood Sugar > 120 mg/dl')
    with col1:
        restecg = st.text_input('Resting Electrocardiographic results')
    with col2:
        thalach = st.text_input('Maximum Heart Rate achieved')
    with col3:
        exang = st.text_input('Exercise Induced Angina')
    with col1:
        oldpeak = st.text_input('ST depression induced by exercise')
    with col2:
        slope = st.text_input('Slope of the peak exercise ST segment')
    with col3:
        ca = st.text_input('Major vessels colored by flourosopy')
    with col1:
        thal = st.text_input('thal: 0 = normal; 1 = fixed defect; 2 = reversable defect')

    if st.button('Heart Disease Test Result'):
        user_input = [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]
        user_input = [float(x) for x in user_input]
        heart_prediction = heart_disease_model.predict([user_input])

        if heart_prediction[0] == 1:
            st.success('The person is having heart disease')
        else:
            st.success('The person does not have any heart disease')

# Pneumonia Prediction Page
if selected == 'Pneumonia Prediction':
    st.title('Pneumonia Prediction using ML')
    st.write("Upload an X-ray image to predict if the person has pneumonia or not.")

    # Input for image upload
    uploaded_image = st.file_uploader("Choose an X-ray image...", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        st.write("Classifying...")

        # Predicting the class of the uploaded image
        pneumonia_prediction = predict(image)
        st.success(f'Prediction: {pneumonia_prediction}')
