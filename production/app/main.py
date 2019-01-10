from flask import Flask, make_response, request
from flask import Response, Flask, render_template
from flask import jsonify,json,request
from flask_cors import CORS

import json
import scipy.stats
import numpy as np
from math import exp, pi, sqrt
import tensorflow as tf
from tensorflow import keras
from utils import *
from keras.models import load_model

def get_score(x):
    
    y = (x - 0.5) / 0.1
    score = scipy.stats.norm(0, 1).pdf(y)/(1/sqrt(2*pi))*100

    return score

global version 
version = 0

global graph
graph = tf.get_default_graph()


global model
model_path = "./models/model_v" + str(version) + ".h5"
model = load_model(model_path)

app = Flask(__name__)
CORS(app)

# global graph,model
# graph = tf.get_default_graph()

@app.route('/')
def form():
    return render_template("index.html")


@app.route('/batch_predict' , methods = ['POST'])
def batch_predict():

    json_string = request.get_json()

    ######################## Input Format ######################## 
    # a list of full image paths
    #
    # {
    #     data: img_paths
    # }

    ######################## Example ########################
    #
    # data_home = "./datasets/retrain/"
    # img_files = ['blue_2357_Brick_corner_1x2x2/201706171206-0009.png', 
    #             'blue_2357_Brick_corner_1x2x2/201706171206-0012.png',
    #             'gray_3023_Plate_1x2/0381.png',
    #             'green_3005_Brick_1x1/0265.png']
    # img_paths = [data_home + img for img in img_files]

    data_home = "./datasets/retrain/"
    img_files = json_string["data"]
    img_paths = [data_home + img for img in img_files]

    x_data =  np.array([ load_image(img_path) for img_path in img_paths])

    classes_decoder = json.load(open("classes_decoder.json"))

    
    with graph.as_default():
        
        probs_pred = model.predict_proba(x_data)
        class_id_pred    = model.predict_classes(x_data)
        class_label_pred = [classes_decoder[str(i)] for i in class_id_pred]
        

    ######################## Output Example ########################
    # first key is the image full path
    # {
    #     './datasets/retrain/blue_2357_Brick_corner_1x2x2/201706171206-0009.png': {
    #                     label: 'gray_3023_Plate_1x2', 
    #                     score: 0.6
    #                   },
    #     './datasets/retrain/blue_2357_Brick_corner_1x2x2/201706171206-0012.png': {
    #                     label: 'gray_3023_Plate_1x2', 
    #                     score: 0.6
    #                   },
    # }

    output = dict()

    for i, class_id in enumerate(class_id_pred):
        
        img_path = img_paths[i]
        output[img_path] = dict()
        
        x = probs_pred[i][class_id]
        
        output[img_path]['label'] = class_label_pred[i]
        output[img_path]['score'] = get_score(x)
        

    return jsonify(output)


@app.route('/update_model', methods=["POST"])
def update_model():

    training_data = request.get_json()
 
    # input: 
    # {
    #     image full path: label
    # }

    # example:
    # {
    #     './datasets/retrain/blue_2357_Brick_corner_1x2x2/201706171206-0009.png': 'blue_2357_Brick_corner_1x2x2',
    #     './datasets/retrain/blue_2357_Brick_corner_1x2x2/201706171206-0012.png': 'blue_2357_Brick_corner_1x2x2',
    #     ...
    # }

    # data_home = "./datasets/retrain/"
    # img_files = ['blue_2357_Brick_corner_1x2x2/201706171206-0009.png', 
    #             'blue_2357_Brick_corner_1x2x2/201706171206-0012.png',
    #             'gray_3023_Plate_1x2/0381.png',
    #             'green_3005_Brick_1x1/0265.png']

    # img_paths = [data_home + img for img in img_files]
    # training_data = { f: f.split("/")[-2] for f in img_paths}

    x_data =  np.array([ load_image(img_path) for img_path in list(training_data.keys())])
    y_data =  list(training_data.values())

    classes_encoder = json.load(open("classes_encoder.json"))
    classes=  list(classes_encoder.keys())
    y_data = np.array([classes_encoder[y] for y in y_data])
    y_data =  convert_to_one_hot(y_data, len(classes)).T

    with graph.as_default():
        
        # retrain
        model.fit(  x_data,
                    y_data,
                    epochs=1,
                    batch_size=16,
                    verbose = 1
                )

        global version
        version = version + 1
        model_path_new = "./models/model_v" + str(version) + ".h5"
        model.save(model_path_new)


    # output:
    # a message confirm model is updated
    output = "Model is updated."
        
    return output


@app.route('/select_version' , methods = ['GET','POST'])
def select_model():

    if request.method == "POST":
        model_version =  request.args['version'] 

        #global graph, model
        global graph, model
        with graph.as_default():
            graph = tf.get_default_graph()
            model_path = "./models/model_v" + str(model_version) + ".h5"
            try:
                model  = keras.models.load_model(model_path)
                output = str(model_path)
            except:
                output = "version not found."

        return jsonify(output)
    
    return render_template("select_model.html")


if __name__ == '__main__':
    app.run(debug=False, port = 8090, host='0.0.0.0')