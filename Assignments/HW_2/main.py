import werkzeug
werkzeug.cached_property = werkzeug.utils.cached_property

import firebase_admin
from firebase_admin import credentials, firestore, db

from keras.preprocessing.image import img_to_array
from keras.models import load_model
from flask_restplus import Api, Resource, fields
from flask import Flask, request, jsonify
import numpy as np
import datetime as dt
from werkzeug.datastructures import FileStorage
from PIL import Image
from keras.models import model_from_json
import tensorflow as tf

cred = credentials.Certificate('./ServiceAccountKey.json')
firebase_app = firebase_admin.initialize_app(cred)
db = firestore.client()

# ref = db.reference('/')
# ref.set({

#         'Employee':
#             {
#                 'emp1':
#                 {
#                     'name':'Subal',
#                     'lname':'Pant',
#                     'age':22
#                 }
#             }
# })

application = app = Flask(__name__)
api = Api(app, version='1.0', title='MNIST Classification', description='CNN for Mnist')
ns = api.namespace('Make_School', description='Methods')

single_parser = api.parser()  # parsing args is one of the benefits of Flask-RESTPlus
single_parser.add_argument('file', location='files',
                           type=FileStorage, required=True)

"""
model = load_model('model_weights.h5')
graph = tf.get_default_graph()
"""

# Model reconstruction from JSON file - do this only once!
# not in the body of a function
with open('model_architecture.json', 'r') as f:
    model = model_from_json(f.read())

# Load weights into the new model
model.load_weights('my_model.h5')
graph = tf.compat.v1.get_default_graph()


@ns.route('/prediction')
class CNNPrediction(Resource):
    """Uploads your data to the CNN"""
    @api.doc(parser=single_parser, description='Upload an mnist image')
    def post(self):
        args = single_parser.parse_args()
        image_file = args.file  # reading args from file
        image_file.save('posted_img.png')  # save the file
        img = Image.open('posted_img.png')  # open the image
        image_red = img.resize((28, 28))  # reshape for the model dimension requirements
        image = img_to_array(image_red)
        print(image.shape)
        x = image.reshape(1, 28, 28, 1)  # 1 image of 28x28 pixels, 1 channel (grayscale)
        x = x/255  # data normalization
        
        with graph.as_default():
            out = model.predict(x)
        print(out[0])
        print(np.argmax(out[0]))
        r = str(np.argmax(out[0]))

        date = str(dt.datetime.now())
        data = {
            u'date': date,
            u'file_name': u'posted_img.png',
            u'result': r 
        }

        db.collection(u'predictions').document(date).set(data)

        return {'prediction': r}


if __name__ == '__main__':
    app.run(host='localhost', port=5000, debug=True)