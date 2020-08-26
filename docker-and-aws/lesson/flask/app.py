import werkzeug
werkzeug.cached_property = werkzeug.utils.cached_property

# Import libraries
from keras.preprocessing.image import img_to_array
from keras.models import load_model
from flask_restplus import Api, Resource, fields
from flask import Flask, request, jsonify
import numpy as np
from werkzeug.datastructures import FileStorage
from PIL import Image
import tensorflow as tf


# Define app and API
app = Flask(__name__)
api = Api(app, version='1.0', title='MNIST Classification', description='CNN for Mnist')
ns = api.namespace('cnn', description='Methods')

# Define parser
single_parser = api.parser()
single_parser.add_argument('file', location='files', type=FileStorage, required=True)

# Load model and define tensorflow graph
model = load_model('my_model.h5')
graph = tf.get_default_graph()

@ns.route('/prediction')
class CNNPrediction(Resource):
    """Uploads your data to the CNN"""
    @api.doc(parser=single_parser, description='Upload an mnist image')
    def post(self):
        # Parse args
        args = single_parser.parse_args()
        image_file = args.file

        # Save file and open it again
        image_file.save('mnist.png')
        img = Image.open('mnist.png')

        # Resize and convert to array
        image_red = img.resize((28, 28))
        image = img_to_array(image_red)
        print(f"Image shape: {image.shape}")

        # Reshape and scale the image array
        x = image.reshape(1, 28, 28, 1)
        x = x/255

        # Make the prediction
        with graph.as_default():
            out = model.predict(x)

        # Round
        print(f"Prediction: {out[0]}")
        print(f"Argmax of prediction: {np.argmax(out[0])}")
        r = np.argmax(out[0])

        # Return prediction as json
        return {'prediction': str(r)}

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)