import  streamlit as st
import numpy as np
#import tensorflow as tf
from sklearn.preprocessing import StandardScaler , LabelEncoder, OneHotEncoder
import pandas as pd
import pickle


#load trained model



model=tensorflow.keras.models.load_model('model.h5')

# Load encoder and scaler
with open('onehot_encode_geo.pkl', 'rb') as file:
    lable_encoder_geo=pickle.load(file)
    
with open('label_encoder_gender.pkl', 'rb') as file:
    lable_encoder_gender=pickle.load(file)
    
with open('scaler.pkl', 'rb') as file:
    scaler=pickle.load(file)
    
## steamlit app
st.title('Customer churn prediction')


# user inputs 
geography=st.selectbox('Geography', lable_encoder_geo.categories_[0])
gender = st.selectbox('Gender', lable_encoder_gender.classes_)
age=st.slider('Age', 18, 92)
balance=st.number_input('Balance')
credit_score=st.number_input('Credit score', 0, 900)
estimated_salary=st.number_input('Estimated salary')
tenure=st.slider('Tenure', 0, 10)
number_of_products=st.slider('No of products', 1, 4)
has_cr_card=st.selectbox('Has credit card', [0, 1])
is_active_member=st.selectbox('Is active member', [0, 1])

# prepare user input data
input_data= pd.DataFrame({
    'CreditScore':[credit_score],
    'Gender':[lable_encoder_gender.transform([gender])[0]],
    'Age':[age],
    'Tenure':[tenure],
    'Balance':[balance ],
    'NumOfProducts':[number_of_products],
    'HasCrCard':[has_cr_card],
    'IsActiveMember':[is_active_member],
    'EstimatedSalary':[estimated_salary]
})


geo_encoded = lable_encoder_geo.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=lable_encoder_geo.get_feature_names_out(['Geography']))

input_data=pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)


# scalling

input_data_scaled=scaler.transform(input_data)


# predict

prediction=model.predict(input_data_scaled)
prediction_prob=prediction[0][0]


if prediction_prob> 0.5:
    st.write('The customer is likely to churn')
else:

    st.write('The customer is not likely to churn')
