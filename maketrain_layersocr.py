"""Convolutional Neural Network Estimator for MNIST, built with tf.layers."""
# Based on https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/tutorials/layers/cnn_mnist.py

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import tempfile
import os
import datetime

import numpy as np
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)

def cnn_model_fn(features, labels, mode):
    """Model function for CNN."""
    # downsizing because of the two convolutions
    img_dim = 4*12 # 48must be divisible by 4
    kernel_length = 10 # 5 before
    
    img_dim_dwn = int(img_dim/2/2)
    
    # Input Layer
    # Reshape X to 4-D tensor: [batch_size, width, height, channels]
    # Input images are img_dimximg_dim pixels, and have one color channel
    
    input_layer = tf.reshape(features["x"], [-1, img_dim, img_dim, 1])

    # Convolutional Layer #1
    # Computes 32 features using a 5x5 filter with ReLU activation.
    # Padding is added to preserve width and height.
    # Input Tensor Shape: [batch_size, img_dim, img_dim, 1]
    # Output Tensor Shape: [batch_size, img_dim, img_dim, 32]
    conv1 = tf.layers.conv2d(
      inputs=input_layer,
      filters=32,
      kernel_size=[kernel_length, kernel_length],
      padding="same",
      activation=tf.nn.relu,
      name='conv1')

    # Pooling Layer #1
    # First max pooling layer with a 2x2 filter and stride of 2
    # Input Tensor Shape: [batch_size, img_dim, img_dim, 32]
    # Output Tensor Shape: [batch_size, img_dim/2, img_dim/2, 32]
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2, name='pool1')

    # Convolutional Layer #2
    # Computes 64 features using a 5x5 filter.
    # Padding is added to preserve width and height.
    # Input Tensor Shape: [batch_size, img_dim/2, img_dim/2, 32]
    # Output Tensor Shape: [batch_size, img_dim/2, img_dim/2, 64]
    conv2 = tf.layers.conv2d(
      inputs=pool1,
      filters=64,
      kernel_size=[kernel_length, kernel_length],
      padding="same",
      activation=tf.nn.relu,
      name='conv2')

    # Pooling Layer #2
    # Second max pooling layer with a 2x2 filter and stride of 2
    # Input Tensor Shape: [batch_size, img_dim/2, img_dim/2, 64]
    # Output Tensor Shape: [batch_size, img_dim/2/2, img_dim/2/2, 64]
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2, name='pool2')

    # Flatten tensor into a batch of vectors
    # Input Tensor Shape: [batch_size, img_dim/2/2, img_dim/2/2, 64]
    # Output Tensor Shape: [batch_size, img_dim/2/2 * img_dim/2/2 * 64]
    pool2_flat = tf.reshape(pool2, [-1, img_dim_dwn * img_dim_dwn * 64], name='pool2_flat')

    # Dense Layer
    # Densely connected layer with 1024 neurons
    # Input Tensor Shape: [batch_size, img_dim/2/2 * img_dim/2/2 * 64]
    # Output Tensor Shape: [batch_size, 1024]
    dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation= tf.nn.relu, name='dense')

    # Add dropout operation; 0.6 probability that element will be kept
    dropout = tf.layers.dropout(inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

    # Logits layer
    # Input Tensor Shape: [batch_size, 1024]
    # Output Tensor Shape: [batch_size, 36]
    logits = tf.layers.dense(inputs=dropout, units=37)

    predictions = {
      # Generate predictions (for PREDICT and EVAL mode)
      "classes": tf.argmax(input=logits, axis=1),
      # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
      # `logging_hook`.
      "probabilities": tf.nn.softmax(logits, name="softmax_tensor", axis=1)
    }
    if mode == tf.estimator.ModeKeys.PREDICT:
        export_outputs = {'predict_output': tf.estimator.export.PredictOutput(predictions)} # https://stackoverflow.com/questions/45640951/tensorflow-classifier-export-savedmodel-beginner?rq=1
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions, export_outputs=export_outputs)
    
    # Calculate Loss (for both TRAIN and EVAL modes)
    #https://www.tensorflow.org/api_docs/python/tf/losses/sparse_softmax_cross_entropy
    #     labels: Tensor of shape [d_0, d_1, ..., d_{r-1}] (where r is rank of labels and result) and dtype int32 or int64. Each entry in labels must be an index in [0, num_classes). Other values will raise an exception when this op is run on CPU, and return NaN for corresponding loss and gradient rows on GPU.
    #     logits: Unscaled log probabilities of shape [d_0, d_1, ..., d_{r-1}, num_classes] and dtype float32 or float64.
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits) # softmax_cross_entropy_with_logits sparse_softmax_cross_entropy. # loss = tf.losses.sparse_softmax_cross_entropy(labels=tf.expand_dims(labels, 2), logits=tf.expand_dims(logits, 2)) # softmax_cross_entropy_with_logits sparse_softmax_cross_entropy sparse_softmax_cross_entropy_with_logits # loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=logits) # softmax_cross_entropy_with_logits sparse_softmax_cross_entropy sparse_softmax_cross_entropy_with_logits
    
    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer(1e-4)
        train_op = optimizer.minimize(loss=loss,global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {"accuracy": tf.metrics.accuracy(labels=labels, predictions=predictions["classes"])}
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


"""
# https://stackoverflow.com/questions/44460362/how-to-save-estimator-in-tensorflow-for-later-use
#https://github.com/tensorflow/tensorflow/issues/12508
#https://www.tensorflow.org/programmers_guide/saved_model#using_savedmodel_with_estimators
def serving_input_receiver_fn():
    #An input receiver that expects a serialized tf.Example
    serialized_tf_example = tf.placeholder(dtype=tf.string,
                                         shape=[default_batch_size],
                                         name='input_example_tensor')
    receiver_tensors = {'examples': serialized_tf_example}
    features = tf.parse_example(serialized_tf_example, feature_spec)
    return tf.estimator.export.ServingInputReceiver(features, receiver_tensors)
   
    
    #An input receiver that expects a serialized tf.Example
    #serialized_tf_example = tf.placeholder(dtype=tf.string,
                                         #shape=[None],
                                         #name='input_tensors')
    #feature_spec = {'x': tf.FixedLenFeature, , tf.estimator.ModeKeys.PREDICT}
    #receiver_tensors = {'predictor_inputs': serialized_tf_example}
    #features = tf.parse_example(serialized_tf_example, feature_spec)
    #return tf.estimator.export.ServingInputReceiver(features, receiver_tensors)
    """
    
def main(unused_argv):
    
    train_dir = os.path.join(os.path.abspath(os.path.join(os.path.realpath(__file__), os.pardir)),'training_set') # '/home/victor/Travail/Python_Projects/coin_OCR/training_set'
    source_images='coinletter_images.npy'
    source_labels='coinletter_labels.npy'
    source_labels_indexes='coinletter_labels_indexes.npy'
    
    # Load training and eval data
    #    labels: Tensor of shape [d_0, d_1, ..., d_{r-1}] (where r is rank of labels and result) and dtype int32 or int64. Each entry in labels must be an index in [0, num_classes). Other values will raise an exception when this op is run on CPU, and return NaN for corresponding loss and gradient rows on GPU.
    #    logits: Unscaled log probabilities of shape [d_0, d_1, ..., d_{r-1}, num_classes] and dtype float32 or float64.
    train_data = np.load(os.path.join(train_dir,source_images)) # extract_images(os.path.join(train_dir,source_images))
    train_labels_indexes = np.asarray(np.load(os.path.join(train_dir,source_labels_indexes)), dtype=np.int32) # extract_labels(os.path.join(train_dir,source_labels)) # , one_hot=one_hot)

    img_dim   = np.shape(train_data)[1]
    nb_labels = 37 # alphabet + numbers + space
    
    eval_data = train_data[:10000]# The TensorFlow documentation clearly states that "labels vector must provide a single specific index for the true class for each row of logits". So your labels vector must include only class-indices like 0,1,2 and not their respective one-hot-encodings like [1,0,0], [0,1,0], [0,0,1]. 
    eval_labels = np.asarray(train_labels_indexes[:10000], dtype=np.int32) # np.asarray(train_labels[:5000], dtype=np.int32)


    # Create the Estimator
    coinOCR_classifier = tf.estimator.Estimator(model_fn=cnn_model_fn, model_dir="/tmp/coinOCR_convnet_model")

    # Set up logging for predictions
    # Log the values in the "Softmax" tensor with label "probabilities"
    tensors_to_log = {"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=50)

    # Train the model
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": train_data},
      y=train_labels_indexes,
      batch_size=100, # 
      num_epochs=None,
      shuffle=True)
    coinOCR_classifier.train(
      input_fn=train_input_fn,
      steps=30000, # max 10000
      hooks=[logging_hook])

    # Evaluate the model and print results
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": eval_data},
      y=eval_labels,
      num_epochs=1,
      shuffle=False)
    eval_results = coinOCR_classifier.evaluate(input_fn=eval_input_fn)
    source_labels
    # Export the model in SavedModel format. https://github.com/tensorflow/tensorflow/blob/abccb5d3cb45da0d8703b526776883df2f575c87/tensorflow/docs_src/programmers_guide/saved_model.md
    #https://www.tensorflow.org/programmers_guide/saved_model#using_savedmodel_with_estimators
    # For saving the model : https://www.tensorflow.org/api_guides/python/meta_graph. Google tutorial : https://www.tensorflow.org/programmers_guide/saved_model#overview_of_saving_and_restoring_models
    # Export inference model :  https://github.com/tensorflow/serving/blob/master/tensorflow_serving/example/inception_saved_model.py and https://github.com/tensorflow/serving/issues/363
    # https://github.com/MtDersvan/tf_playground/blob/master/wide_and_deep_tutorial/wide_and_deep_basic_serving.md
    #serialized_tf_example = tf.placeholder(dtype=tf.string, shape=[None], name='input_tensors')
    #receiver_tensors      = {"predictor_inputs": serialized_tf_example}
    #feature_spec          = {"words": tf.FixedLenFeature([25],tf.int64)}
    #features              = tf.parse_example(serialized_tf_example, feature_spec)
    #return tf.estimator.export.ServingInputReceiver(features, receiver_tensors)
    #feature_spec = {'x': tf.FixedLenFeature(dtype=tf.float32, shape=[-1, img_dim, img_dim, 1])} # feature is the input

    export_dir = os.path.join(os.path.join(os.path.abspath(os.path.join(os.path.realpath(__file__), os.pardir)),'TF_SavedModel/'),'TF_SavedModel_'+datetime.datetime.now().strftime('%Y-%m-%d-%H-%M')) # ('/home/victor/Travail/Python_Projects/TF_SavedModel'
    print('Exporting trained model to %s' % export_dir)
    
    # DEfining th signature https://www.tensorflow.org/serving/signature_defs
    
    img_dim = 48
    feature_inputs = {"x": tf.placeholder(dtype=np.float32, shape=[1, img_dim, img_dim, 1])} # , labels
    input_receiver_fn = tf.estimator.export.build_raw_serving_input_receiver_fn(feature_inputs) # tf.estimator.build_raw_serving_input_receiver_fn()
        
    export_location = coinOCR_classifier.export_savedmodel(export_dir, input_receiver_fn) #export_location = tf.estimator.export_savedmodel(export_dir, input_receiver_fn)
        
    print('Successfully exported model to %s' % export_dir)

if __name__ == "__main__":
    tf.app.run()