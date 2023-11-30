## import the libraries
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from sklearn.impute import SimpleImputer
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
from PIL import Image
from sklearn.linear_model import LogisticRegression
from keras.models import load_model


# Load the model from the file
bcan = pd.read_csv('Breast_Cancer.csv')

from tensorflow.keras.models import load_model


## Load the ANN model
Breast_cancer_Annmodel = load_model('Breast_cancer_Annmodel.h5')

## creat a function of prediction, we name it BCAN
def BCAN_prediction(input):
    input_array = np.asarray(input)
    input_reshape = input_array.reshape(1,-1)
    
    
    #log_reg = LogisticRegression()
    #prediction = log_reg.predict(input_reshape)
    
    prediction = Breast_cancer_Annmodel.predict(input_reshape)
    print(prediction)
    
    # predict---I need to write a code here to say small number is more chance to survive
    if (prediction[0] ==0):
        return 'Your health conditions indicate a lower risk of a negative outcome related to breast cancer.'# 0
    else: 
        return 'Your health conditions suggest a higher chance of a negative outcome related to breast cancer.' # 1
    
def main():
    ## set page configuration 
    st.set_page_config(page_title='Breast Cancer Predictor', layout='wide')

    ## add image
    image = Image.open(r"C:\Users\PC\Desktop\Final Project\BCAN.png")
    st.image(image, use_column_width=False)

    ## add page title and content
    st.title('Breast Cancer Survival Predictor Using Artificial Neural Networks')
    st.write('Enter your personal data to get Breast Cancer risk evaluation')
    
    ## variable inputs
    Age = st.number_input('Age of the patient:',min_value=0, step=1)
    Race = st.number_input('Race | Blck = 0, Other= 1, White=2:',min_value=0, step=1)
    Marital_Status = st.number_input('Marital Status)| Divorced=0, Married=1,Separated = 2, Single =3, Widowed =4:',min_value=0, step=1)
    Sixth_Stage = st.number_input('6th Stage| Enter the corresponding number "IIA": 0, "IIB": 1, "IIIA": 2, "IIIB": 3, "IIIC": 4:',min_value=0, step=1)
    differentiate = st.number_input('differentiated | Moderately differentiated: 0, Poorly differentiated: 1, Undifferentiated : 2, Well differentiated: 3:',min_value=0, step=1)
    A_Stage = st.number_input('A_Stage| Enter the corresponding number (0 for Distant, 1 for Regional):', min_value=0, max_value=3, step=1)
    Tumor_Size = st.number_input('Tumor Size|Enter the corresponding number (1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 8: 7, 9: 8, 10: 9, 11: 10, 12: 11, 13: 12, 14: 13, 15: 14, 16: 15, 17: 16, 18: 17, 19: 18, 20: 19, 21: 20, 22: 21, 23: 22, 24: 23, 25: 24, 26: 25, 27: 26, 28: 27, 29: 28, 30: 29, 31: 30, 32: 31, 33: 32, 34: 33, 35: 34, 36: 35, 37: 36, 38: 37, 39: 38, 40: 39, 41: 40, 42: 41, 43: 42, 44: 43, 45: 44, 46: 45, 47: 46, 48: 47, 49: 48, 50: 49, 51: 50, 52: 51, 53: 52, 54: 53, 55: 54, 56: 55, 57: 56, 58: 57, 59: 58, 60: 59, 61: 60, 62: 61, 63: 62, 64: 63, 65: 64, 66: 65, 67: 66, 68: 67, 69: 68, 70: 69, 72: 70, 73: 71, 74: 72, 75: 73, 76: 74, 77: 75, 78: 76, 79: 77, 80: 78, 81: 79, 82: 80, 83: 81, 84: 82, 85: 83, 86: 84, 87: 85, 88: 86, 90: 87, 92: 88, 94: 89, 95: 90, 96: 91, 97: 92, 98: 93, 100: 94, 101: 95, 103: 96, 104: 97, 105: 98, 107: 99, 108: 100, 110: 101, 115: 102, 117: 103, 120: 104, 123: 105, 125: 106, 130: 107, 133: 108, 140: 109) :',min_value=0, step=1)
    Estrogen_Status = st.number_input('Estrogen_ Status|Positive = 1, Negative = 0:',min_value=0, step=1)
    Progesterone_Status = st.number_input('Progestron|Positive = 1, Negative= 0:',min_value=0, step=1)
    N_Stage = st.number_input('N_Stage| Enter the corresponding number (0 for N1, 1 for N2, 2 for N3):', min_value=0, max_value=3, step=1)
    T_Stage = st.number_input('Tumor Stage (T_Stage): Enter the corresponding number (0 for T1, 1 for T2, 2 for T3, 3 for T4):', min_value=0, max_value=3, step=1)
    Grade = st.number_input('Grade | anaplastic; Grade IV: 0, 1: 1, 2: 2, 3: 3 :',min_value=0, step=1)
    Regional_Node_Examined = st.number_input('Enter the value of Regional_Node_Examined; from 1 to 61:',min_value=0, step=1)
    Regional_Node_Positive = st.number_input('Enter the value of Regional_Node_Positive; from 1 to 46:',min_value=0, step=1)
    Survival_Months = st.number_input('Enter the value of Survival_Months:',min_value=0, step=1)
   

    ## code for prediction
    predict = ''

    # button for prediction
    if st.button('Predict'):
        predict = BCAN_prediction([Age, Race, Marital_Status,Sixth_Stage, T_Stage , N_Stage, differentiate, Grade, A_Stage, Tumor_Size, Estrogen_Status,Regional_Node_Examined, Progesterone_Status,Survival_Months, Regional_Node_Positive,  ])

    st.success(predict)

if __name__ == '__main__':
    main()
    
    
    
    
    
    
    ##command in Terminal 
    # cd "C:\Users\PC\Desktop\Final Project" 
    # streamlit run BC.py