from flask import Flask, render_template, request, jsonify
import pandas as pd
import pickle
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import PolynomialFeatures

app = Flask(__name__)

# Load the model and encoder
with open('model/improved_weather_temperature_model.pkl', 'rb') as model_file:
    rf_model = pickle.load(model_file)

# Initialize the encoder (make sure it matches the one used for training)
encoder = OneHotEncoder(handle_unknown='ignore')
encoder.fit([['District1'], ['District2'], ['District3']])  # Example districts

# Initialize polynomial features (use the same parameters as used in training)
poly = PolynomialFeatures(degree=2, include_bias=False)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    district = data['district']
    date = pd.to_datetime(data['date'], format='%Y-%m-%d')
    month = date.month
    day = date.day

    # One-Hot Encode District
    district_encoded = encoder.transform([[district]])
    features = pd.DataFrame(district_encoded.toarray(), columns=encoder.get_feature_names_out(['District']))

    # Add polynomial features for Month and Day
    time_features = pd.DataFrame({'Month': [month], 'Day': [day]})
    time_poly = poly.fit_transform(time_features)
    time_poly_df = pd.DataFrame(time_poly, columns=poly.get_feature_names_out(['Month', 'Day']))

    # Concatenate all features
    features = pd.concat([features.reset_index(drop=True), time_poly_df], axis=1)

    # Add scaled features (if needed, adjust based on your model)
    features['Min Humidity (%)'] = 0
    features['Max Humidity (%)'] = 0
    features['Min Wind Speed (Kmph)'] = 0
    features['Max Wind Speed (Kmph)'] = 0

    # Predict using the improved model
    prediction = rf_model.predict(features)

    return jsonify({'average_temperature': prediction[0]})

if __name__ == '__main__':
    app.run(debug=True)
