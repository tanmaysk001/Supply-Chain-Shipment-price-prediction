import streamlit as st
import pandas as pd
import numpy as np
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import os

st.set_page_config(page_title='Webapp for machine learning model', 
                    page_icon=':chart_with_upwards_trend:',
                    layout="wide",
                    initial_sidebar_state="expanded")


model = joblib.load('Shipment cost prediction project/xgb_model_5.json')

def predict_shipment_price(line_item_quantity, freight_cost, weight, pack_price, unit_price):
    input_data = np.array([[line_item_quantity, freight_cost, weight, pack_price, unit_price]])
    #scaler = MinMaxScaler()
    #input_data_scaled = scaler.fit_transform(input_data)
    prediction = model.predict(input_data)
    return prediction[0]

def main():
    st.title("Shipment Price Prediction")
    st.text('In this webapp, I have deployed a ML model which can predict shipment prices according to the following input parameters. \nThe dataset that I have used for this project is available on kaggle.com')

    #sliders
    line_item_quantity = st.slider("Line Item Quantity", min_value = 1.0, max_value = 15.0, step = 0.05)
    freight_cost = st.slider("Freight Cost (USD)", min_value = 7.0, max_value = 14.0 , step = 0.05)
    weight = st.slider("Weight (Kilograms)", min_value = 4.0, max_value = 14.0, step = 0.05)
    pack_price = st.slider("Pack Price", min_value = 0.0, max_value = 7.0, step = 0.05)
    unit_price = st.slider("Unit Price", min_value = 0.0, max_value = 2.0, step = 0.05)
    if st.button("Predict"):

        if line_item_quantity <= 0 or freight_cost <= 0 or weight <= 0 or pack_price < 0 or unit_price < 0:
            st.warning("Please enter valid input values.")
            return

    # Make prediction and display results
        try:
            prediction = predict_shipment_price(line_item_quantity, freight_cost, weight, pack_price, unit_price)
            st.success("The predicted shipment price is $ {}".format(round(prediction, 2)))

            # Save prediction to a CSV file
            df = pd.DataFrame({
                'Line Item Quantity': [line_item_quantity],
                'Freight Cost (USD)': [freight_cost],
                'Weight (Kilograms)': [weight],
                'Pack Price': [pack_price],
                'Unit Price': [unit_price],
                'Predicted Shipment Price': [prediction]
            })
            df.to_csv('Generated Data/predictions.csv', mode='a', index=False, header=not os.path.isfile('Generated Data/predictions.csv'))

            # Reset sliders to default values
            line_item_quantity = 1.0
            freight_cost = 7.0
            weight = 4.0
            pack_price = 0.0
            unit_price = 0.0
        except Exception as e:
            st.error(f"An error occurred while making the prediction: {e}")

if __name__ == "__main__":
    main()    