## Import Libraries
import streamlit as st
import joblib
import numpy as np
from utlis import process_new


## Load the model
model_svc = joblib.load('svc_important.pkl')




def churn_classification():

    ## Title
    st.title('Telecom Churn Classification Prediction ....')
    st.markdown('<hr>', unsafe_allow_html=True)



    ## Input fields

    contract = st.selectbox('contract', options=['Month-to-month', 'One year', 'Two year'])
    tenure = st.number_input('tenure', value=20, step=1)
    monthlycharges = st.number_input('monthlycharges', value=100)
    totalcharges = st.number_input('totalcharges', value=50)
    techsupport = st.selectbox('techsupport', options=['No', 'Yes',"No internet service"])
    onlinesecurity = st.selectbox('onlinesecurity', options=['No', 'Yes',"No internet service"])
    paperlessbilling = st.selectbox('paperlessbilling', options=['No', 'Yes'])
    paymentmethod = st.selectbox('paymentmethod', options=['Electronic check', 'Mailed check',"Bank transfer (automatic)","Credit card (automatic)"])
    internetservice = st.selectbox('internetservice', options=['DSL', 'Fiber optic',"No"])
    dependents = st.selectbox('dependents', options=['No', 'Yes'])      
   

    st.markdown('<hr>', unsafe_allow_html=True)


    if st.button('Predict Churn ...'):

        ## Concatenate the users data
        new_data = np.array([contract, tenure, monthlycharges, totalcharges, techsupport,
                            onlinesecurity, paperlessbilling, paymentmethod, internetservice,dependents])
        
        ## Call the function from utils.py to apply the pipeline
        X_processed = process_new(X_new=new_data)

        ## Predict using Model
        
        y_pred = model_svc.predict(X_processed)[0]


        y_pred = bool(y_pred)

        ## Display Results
        st.success(f'Churn Prediction is ... {y_pred}')



if __name__ == '__main__':
    ## Call the function
    churn_classification()

