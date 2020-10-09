# Make a flask API for our DL Model
import werkzeug
werkzeug.cached_property = werkzeug.utils.cached_property

from keras.preprocessing.image import img_to_array
from keras.models import load_model
from flask_restplus import Api, Resource, fields
from flask import Flask, request, jsonify
import numpy as np
from werkzeug.datastructures import FileStorage
from PIL import Image
from keras.models import model_from_json
import tensorflow as tf
from tensorflow.python.keras.backend import set_session

app = Flask(__name__)
api = Api(app, version='1.0', title='MNIST Classification', description='CNN for Mnist')
ns = api.namespace('Make_School', description='Methods')
application = app

single_parser = api.parser()
single_parser.add_argument('file', location='files',
                           type=FileStorage, required=True)

tf.compat.v1.disable_eager_execution()
sess = tf.compat.v1.Session()
model = load_model('mnist_cnn_model.h5')
graph = tf.compat.v1.get_default_graph()

# IMPORTANT: models have to be loaded AFTER SETTING THE SESSION for keras!
# Otherwise, their weights will be unavailable in the threads after the session there has been set
set_session(sess)

# Model reconstruction from JSON file
with open('model_architecture.json', 'r') as f:
    model = model_from_json(f.read())
#
# Load weights into the new model
model.load_weights('model_weights.h5')

@ns.route('/prediction')
class CNNPrediction(Resource):
    """Uploads your data to the CNN"""
    @api.doc(parser=single_parser, description='Upload an mnist image')
    def post(self):
        args = single_parser.parse_args()
        image_file = args.file
        image_file.save('milad.png')
        img = Image.open('milad.png')
        image_red = img.resize((28, 28))
        image = img_to_array(image_red)
        print(image.shape)
        x = image.reshape(1, 28, 28, 1)
        x = x/255
        # This is not good, because this code implies that the model will be
        # loaded each and every time a new request comes in.
        # model = load_model('my_model.h5')
        with graph.as_default():
            set_session(sess)
            out = model.predict(x)
        print(out[0])
        print(np.argmax(out[0]))
        r = np.argmax(out[0])

        return {'prediction': str(r)}


if __name__ == '__main__':
    application.run()