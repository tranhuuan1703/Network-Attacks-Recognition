from flask import Flask, render_template, request
import sys

sys.path.append('./flaskr')
from deploy import *

app = Flask(__name__)

data = np.array([-0.11589662, -0.44763997,  0.73242686,  1.26709163, -0.55699514, -0.38077126,
                         -0.01538439, -0.08961937,  0.,         -0.06391472,  0.   ,      -0.78191504,
                         -0.00833679, -0.02371279, -0.00686246 ,-0.0339577,  -0.0218504,  -0.01025688,
                         -0.04645718,  0. ,        -0.00815811, -0.05426096, -0.75327877, -0.37621105,
                         -0.63088101, -0.92632188,  2.54214737,  2.52402644,  0.79809069, -0.35689433,
                         -0.38294512, -1.50626651, -1.07222966, -1.12303085,  0.34965693, -0.34564847,
                         -0.28633661, -0.49269859, -0.6168307,  -0.3278422,   2.54040052])

result = predict_model(data)


@app.route('/')
def index_view():
    return render_template('index.html')


@app.route('/predict/',methods=['GET','POST'])
def predict():
    response = str(result)
    return response

if __name__ == '__main__':
    app.run(debug=True, port=8000)
