import json, argparse, time, numpy as np, random

import tensorflow as tf
from load import load_graph

from flask import Flask, request
from flask_cors import CORS

import sys
import pickle

##################################################
# API part
##################################################
app = Flask(__name__)
cors = CORS(app)

@app.route("/api/get", methods=['GET'])
def get():
    return "Hello, world!"

api_responses = ["none", "walking", "upstairs", "downstairs", "sitting", "standing", "laying"]

@app.route("/api/activity", methods=['POST'])
def activity():
    clientstr = request.data.decode("utf-8")
    #print(data)
    if clientstr == "":
        params = request.form
        x_in = json.loads(params['x'])
    else:
        params = json.loads(clientstr)
        x_in = params['x']

    data = np.asarray(x_in)
    for reading in data:
        print(reading)
    print("len(model_input): " + str(len(data)))
    print("len(model_input[0]): " + str(len(data[0])))

    # normalize the data by feature to fit in [-1,1]
    """"
    for i in range(9):
        data[:,i] = (data[:,i]-np.min(data[:,i]))/(np.max(data[:,i])-np.min(data[:,i])) # translate everything to above 0
        data[:,i] = data[:,i]*2
        data[:,i] -= 1
    """

    data = normalized(data)
    data = np.reshape(data, (1, 128, 9))  # reformat data into 3D tensor

    # run the model
    feed_dict = {X: data}  # Create a feed_dict with data
    pred = sess.run(Pred_Y, feed_dict=feed_dict)  # make prediction
    classification = np.argmax(pred) + 1  # print result to STDOUT (classes are indexed from 1)
    print(api_responses[classification])

    return json.dumps({"classification": api_responses[classification]})

def normalized(a, axis=-1, order=2):
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2==0] = 1
    return a / np.expand_dims(l2, axis)

if __name__ == "__main__":

    # Load tensorflow model
    global sess, X, Y, Pred_X, Pred_Y

    # Create a clean graph and import the MetaGraphDef nodes
    tf.reset_default_graph()
    sess = tf.Session()
    print("loading meta graph")
    saver = tf.train.import_meta_graph(sys.argv[1] + '.meta')
    print("loaded meta graph")
    saver.restore(sess, sys.argv[1])
    print("restored saver")
    all_vars = tf.get_collection('vars')
    print("got all vars")
    X = all_vars[0]
    Y = all_vars[1]
    Pred_Y = all_vars[2]

    print('Starting the API')
    app.run(host='0.0.0.0', port=80)
