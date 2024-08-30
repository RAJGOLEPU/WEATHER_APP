import pickle

def load_model():
    with open('model/improved_weather_temperature_model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    return model