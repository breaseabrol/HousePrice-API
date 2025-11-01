# House Price Prediction API

# Copyright (c) 2025 Branden Rease Abrol

# Licensed under the MIT License


from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib


app = FastAPI(title="Housing Price Prediction API")

try:
    model = joblib.load("model.pkl")
    scaler = joblib.load("scaler.pkl")
except Exception as e:
    raise RuntimeError(f"Error loading model or scaler: {e}")

feature_order = [
    'area', 'bedrooms', 'bathrooms', 'stories', 'parking',
    'furnishingstatus', 'mainroad_yes', 'guestroom_yes',
    'basement_yes', 'hotwaterheating_yes', 'airconditioning_yes',
    'prefarea_yes', 'area_bedrooms', 'stories_bathrooms',
    'luxury_index','amenities_count'
]

class HouseData(BaseModel):
    area: float
    bedrooms: int
    bathrooms: int
    stories: int
    parking: int
    furnishingstatus: int
    mainroad_yes: int
    guestroom_yes: int
    basement_yes: int
    hotwaterheating_yes: int
    airconditioning_yes: int
    prefarea_yes: int

@app.post("/predict")
def predict(data: HouseData):
    try:
        df = pd.DataFrame([data.dict()])

        # Scale numerical features
        df[['area']] = scaler.transform(df[['area']])

        # Feature engineering
        df['area_bedrooms'] = df['area'] * df['bedrooms']
        df['stories_bathrooms'] = df['stories'] * df['bathrooms']

        amenity_cols = [
            'mainroad_yes', 'guestroom_yes', 'basement_yes',
            'hotwaterheating_yes', 'airconditioning_yes', 'prefarea_yes'
        ]

        df['amenities_count'] = df[amenity_cols].sum(axis=1)
        df['luxury_index'] = (
            df['airconditioning_yes']
            + df['hotwaterheating_yes']
            + df['guestroom_yes']
            + df['basement_yes']
        )

        df = df[feature_order]

        pred = model.predict(df)
        return {"prediction": float(pred[0])}

    except Exception as e:
        return {"error": str(e)}
