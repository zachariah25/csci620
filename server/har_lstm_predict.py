# -*- coding: utf-8 -*-
# -*- built with Python 3.6 -*-
"""
har_lstm_predict.py

Loads in a LSTM saved as a TensorFlow .meta object and then performs a prediction
on incoming data. Outputs a single integer for each 128 time step data instance
provided corresponding to the predicted class. WIll continously query STDIN for 
a new data matrix file containing a single data series instance. Will print prediction
for each vector it receives to STDOUT. Data instance should be pickled numpy arrays
of size 128 x 9 (timesteps x input channels). Feed the filename into STDIN.

Output class correspondances:
    
1 = WALKING
2 = WALKING_UPSTAIRS
3 = WALKING_DOWNSTAIRS
4 = SITTING
5 = STANDING
6 = LAYING

HOW TO RUN:
    
    python har_lstm_predict.py {1}
    
WHERE:
    
    {1} = The name of the TensorFlow .meta graph (ex. "saved_model/har_lstm_graph")

@author: Brody Kutt (bjk4704@rit.edu), Poppy Immel (pgi8114@rit.edu,), Zach Lauzon (zrl3031@rit.edu)
"""

import tensorflow as tf  # built with version 0.12.1
import numpy as np
import sys
import pickle
                

if __name__ == '__main__':
    if(len(sys.argv) == 2):
        
        # Create a clean graph and import the MetaGraphDef nodes
        tf.reset_default_graph()
        with tf.Session() as sess:
          # Import the previously exported meta graph and variables
          saver = tf.train.import_meta_graph(sys.argv[1] + '.meta')
          saver.restore(sess, sys.argv[1])
          all_vars = tf.get_collection('vars')
          X = all_vars[0]
          Y = all_vars[1]
          Pred_Y = all_vars[2]
          # Continuously query stdin for new data instances
          while(True):
              print('Please enter filename...')
              file = input()
              try:
                  data = pickle.load(open(file, "rb" ))
              except FileNotFoundError:
                  print("Wrong file or file path provided.")
                  sys.exit(1)
              
              data = np.reshape(data, (1, 128, 9))  # reformat data into 3D tensor
              feed_dict = {X: data}  # Create a feed_dict with data
              pred = sess.run(Pred_Y, feed_dict=feed_dict)  # make prediction
              print(np.argmax(pred)+1)  # print result to STDOUT (classes are indexed from 1)
    else:
        print('Wrong number of arguments.')
        print("""
        HOW TO RUN:
    
            python har_lstm_predict.py {1}
    
        WHERE:
    
            {1} = The name of the TensorFlow .meta graph ex. "saved_model/cnn_graph"
            """)