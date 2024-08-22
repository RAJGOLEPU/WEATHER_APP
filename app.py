from flask import Flask, render_template, request, jsonify
import model  # Import the dummy model

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json  # Get data from the frontend
    location = data['location']
    
    # Use the dummy model to make predictions
    prediction = model.predict_weather(location)
    
    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    app.run(debug=True)
