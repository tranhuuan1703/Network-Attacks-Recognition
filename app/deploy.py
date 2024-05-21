
import json
import h5py
from keras.models import model_from_json
import numpy as np

# read file json and load model
url_model = "D:\\machine\\model.json"
url_weight = "D:\\machine\\weights.h5"
# create app flask

with open(url_model, "r") as json_file:
    loaded_model_json = json_file.read()

loaded_model = model_from_json(loaded_model_json)

loaded_model.load_weights(url_weight)


def format_input(data_input):
    data = data_input.reshape(-1, len(data_input))
    return data

def predict_model(data_input):

    data = format_input(data_input)
    prediction = loaded_model.predict(data)
    return prediction.item()

data = np.array([-0.11589662, -0.44763997,  0.73242686,  1.26709163, -0.55699514, -0.38077126,
                         -0.01538439, -0.08961937,  0.,         -0.06391472,  0.   ,      -0.78191504,
                         -0.00833679, -0.02371279, -0.00686246 ,-0.0339577,  -0.0218504,  -0.01025688,
                         -0.04645718,  0. ,        -0.00815811, -0.05426096, -0.75327877, -0.37621105,
                         -0.63088101, -0.62632188,  2.54214737,  2.52402644,  0.79809069, -0.35689433,
                         -0.38294512, -1.50626651, -1.07222966, -1.12303085,  0.34965693, -0.34564847,
                         -0.28633661, -0.49269859, -0.6168307,  -0.3278422,   2.54040052])


# result = predict_model(data)
#
# print(type(result))