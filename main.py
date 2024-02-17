import streamlit as st
import joblib



def main(model_path):
    model = joblib.load(model_path)

    # Streamlit UI
    st.title('Sales Prediction App')

    # Input fields for feature values
    tv = st.number_input('TV Advertisements', min_value=0.0, step=1.0)
    radio = st.number_input('Radio Advertisements', min_value=0.0, step=1.0)
    newspaper = st.number_input('Newspaper Advertisements', min_value=0.0, step=1.0)

    # Make prediction when 'Predict' button is clicked
    if st.button('Predict Sales'):
        # Prepare input features as a NumPy array
        features = [[tv, radio, newspaper]]
        
        # Make prediction
        prediction = model.predict(features)
        
        # Display prediction
        st.write(f'\nPredicted Sales: {prediction[0]:.2f}')


if __name__=='__main__':
    # Load the saved model from the file
    model_path = 'model/rf_regressor.joblib'  
    main(model_path)  
