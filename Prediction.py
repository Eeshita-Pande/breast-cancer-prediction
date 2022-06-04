import numpy as np
import pickle 
import streamlit as st
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from PIL import Image

def predictions(x, model_selection):
    
    scaler = StandardScaler(with_mean=True, with_std=True)
    x_norm = scaler.fit_transform(x)

    loaded_model = ''
        
    if model_selection == 'Simple NN':
        loaded_model = tf.keras.models.load_model('Simple_NN', experimental_io_device='/job:localhost')
    elif model_selection == 'Deep NN':
        loaded_model = tf.keras.models.load_model('Deep_NN')
    elif model_selection == 'Decision Tree':
        loaded_model = pickle.load(open('dt_clf.sav', 'rb'))
    elif model_selection == 'Random Forest':
        loaded_model = pickle.load(open('rnd_clf.sav', 'rb'))
    elif model_selection == 'XG Boost':
        loaded_model = pickle.load(open('XGB_clf.sav', 'rb'))
    elif model_selection == 'Support Vector Machines':
        loaded_model = pickle.load(open('SVM_clf.sav', 'rb'))
    elif model_selection == 'Naive Bayes':
        loaded_model = pickle.load(open('gnb.sav', 'rb'))
    elif model_selection == 'KNN':
        loaded_model = pickle.load(open('neigh.sav', 'rb'))
    else:
        print("Enter valid model name")
    

    y = loaded_model.predict(x_norm)
    y = np.where(y > 0.5, 1, 0)
    
    if y == 1:
        return("Likely to be breast cancer")
    else: 
        return("Unlikely to be breast cancer")


def main():
    st. set_page_config(layout="wide") 
    st.title("BI-RADS based Breast Cancer Prediction")
    st.sidebar.header("Methodology")
    st.sidebar.markdown("<h4 style='text-align: justify;'>The Breast Imaging Reporting & Data System (BI-RADS) is a comprehensive assessment system developed by the American College of Radiology to classify breast images. I have used a public dataset from the UCI repository to train several models on predicting breast cancer when presented with inputs for BI-RADS categories (explained below).", unsafe_allow_html=True)
    st.sidebar.header("Age of the Patient")
    st.sidebar.markdown("<h4 style='text-align: justify;'>Patient's age in years (whole numbers).", unsafe_allow_html=True)
    st.sidebar.header("Shape of the Mass")
    st.sidebar.markdown("<h4 style='text-align: justify;'>Masses can be round, oval (with no lobulations or multiple lobulations), or irregular.", unsafe_allow_html=True)
    st.sidebar.header("Margin of the Mass")
    st.sidebar.markdown("<h4 style='text-align: justify;'>Margins can be circulscribed i.e. +75% of circumference is well-defined (usually benign), microlobulated i.e. small undulations (usually suspicious) , obscured i.e. +25% of the circumference is hidden, indistinct i.e. none of the circumference is well-defined (usually suspicious), spiculated i.e. sharp, linear radiations (usually very suspicious).", unsafe_allow_html=True)
    st.sidebar.header("Density of the Mass")
    st.sidebar.markdown("<h4 style='text-align: justify;'>High density masses are usually malignant. Low density and fat containing masses are usually benign.", unsafe_allow_html=True)
    st.sidebar.header("Predictive Model")
    st.sidebar.markdown("<h4 style='text-align: justify;'>List of predictive models trained. Select from dropdown.", unsafe_allow_html=True)
    Age = st.text_input('Age')
    Shape = st.slider("Shape", min_value=1, max_value=4, step=1, value=2)
    st.image('shape.png',width=575,use_column_width='never',output_format='PNG', caption='Round (input:1), oval but non-lobular (input:2), oval and lobular (input:3), irregular (input:4).')
    Margin = st.slider('Margin', min_value=1, max_value=5, step=1, value=2)
    st.image('margin.png',width=1000,use_column_width='never',output_format='PNG', caption='Circumscribed (input:1), microlobulated (input:2) , obscured (input:3), indistinct (input:4), spiculated (input:5).')
    Density = st.slider('Density',  min_value=1, max_value=4, step=1, value=2)
    st.image('density.png',width=790,use_column_width='never',output_format='PNG', caption='High density (input:1), equal density (input:2), low density (input:3), fat containing (input:4)')
    Model = st.selectbox("Predictive Model", ['Simple NN', 'Deep NN', 'Decision Tree', 'Random Forest', 'XG Boost', 'Support Vector Machines', 'Naive Bayes', 'KNN'])
    
    diagnosis = ''
    
    if st.button('Prediction'):
        diagnosis = predictions([[Age, Shape, Margin, Density]], Model) 
    
    st.success(diagnosis)


if __name__== '__main__':
    main()





