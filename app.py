import streamlit as st
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import OrdinalEncoder

reg_model=joblib.load("diamond_model1.pkl")
cluster_model=joblib.load("best_kmean.pkl")

cluster_names ={
    0:"Premium Heavy Diamonds",
    1:"Mid-range Balanced Diamonds",
    2:"Affordable Small Diamonds"
}

st.title("Diamond Price & Market Segment Prediction")

st.header("Enter Diamond Attributes")

carat =st.number_input("Carat", min_value=0.1, max_value=5.0, step=0.01)
x =st.number_input("X dimension (mm)", min_value=0.1, step=0.1)
y =st.number_input("Y dimension (mm)", min_value=0.1, step=0.1)
z =st.number_input("Z dimension (mm)", min_value=0.1, step=0.1)
cut =st.selectbox("Cut", ["Fair", "Good", "Very Good", "Premium", "Ideal"])
color =st.selectbox("Color", ["D","E","F","G","H","I","J"])
clarity =st.selectbox("Clarity", ["I1","SI2","SI1","VS2","VS1","VVS2","VVS1","IF"])

if carat < 0.5:
    carat_category = "Light"
elif carat < 1.0:
    carat_category = "Medium"
else:
    carat_category = "Heavy"

cut_order = ['Fair', 'Good', 'Very Good', 'Premium', 'Ideal']
color_order = ['D', 'E', 'F', 'G', 'H', 'I', 'J']
clarity_order = ['I1', 'SI2', 'SI1', 'VS2', 'VS1', 'VVS2', 'VVS1', 'IF']
carat_order = ['Light', 'Medium', 'Heavy']

ord_enc =OrdinalEncoder(categories=[cut_order, color_order, clarity_order, carat_order])

input_df =pd.DataFrame([{
    "cut":cut,
    "color":color,
    "clarity":clarity,
    "carat_category":carat_category
}])

cut_encoded, color_encoded,clarity_encoded, carat_encoded = ord_enc.fit_transform(input_df)[0]

input_data = pd.DataFrame([{
    "carat": np.sqrt(carat), 
    "x":x,
    "y":y,
    "z":z,
    "cut_encoded":cut_encoded,
    "color_encoded":color_encoded,
    "clarity_encoded":clarity_encoded,
    "carat_encoded":carat_encoded
}])

if st.button("Predict Price"):
    sqrt_price_pred =reg_model.predict(input_data)[0]
    price_pred = sqrt_price_pred**2
    st.success(f"Predicted Diamond Price:₹ {price_pred:,.2f}")

if st.button("Predict Market Segment"):
    cluster_pred = cluster_model.predict(input_data)[0]
    cluster_name = cluster_names.get(cluster_pred, f"Cluster{cluster_pred}")
    st.info(f"Diamond belongs to:{cluster_name} (Cluster{cluster_pred})")
