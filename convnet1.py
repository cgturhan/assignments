# -*- coding: utf-8 -*-
"""
Created on Fri Nov 11 17:22:08 2016

@author: ceren
"""

import tensorflow as tf
import numpy as np
import zipfile
from matplotlib import pyplot as plt

# load dataset
zip_ref = zipfile.ZipFile("data/ORL_faces.npz.zip", 'r')
zip_ref.extractall("data/")
zip_ref.close()

data = np.load('data/ORL_faces.npz')

trainX = data['trainX']
testX = data['testX']
trainY = data['trainY']
testY = data['testY']

# parameters
n_class = np.unique(trainY).size
n_input = trainX.shape[1]
train_size = trainX.shape[0]
test_size = testX.shape[0]
M, N = 112, 92
batch_size = 64
display_step = 10
learning_rate = .01
iters = 10000

# convert labels to one-hot vectors
trainY = np.eye(n_class)[trainY]
testY = np.eye(n_class)[testY]

# tf inputs, variables
x = tf.placeholder(tf.float32, [None, n_input])
y = tf.placeholder(tf.float32, [None, n_class])

weights = {
    # 7x7 conv, 1 input, 32 outputs
    'wconv1': tf.Variable(tf.random_normal([7, 7, 1, 32])),
    # 7x7 conv, 32 inputs, 64 outputs
    'wconv2': tf.Variable(tf.random_normal([7, 7, 32, 64])),
    # 7x7 conv, 64 inputs, 64 outputs
    'wconv3': tf.Variable(tf.random_normal([7, 7, 64, 64])),
    # fully connected, 28*23*64 inputs, 1024 outputs
    'wfc1': tf.Variable(tf.random_normal([28*23*64, 1024])),
    # 1024 inputs, 20 outputs   
    'out': tf.Variable(tf.random_normal([1024, n_class]))
}

biases = {
    'bconv1': tf.Variable(tf.random_normal([32])),
    'bconv2': tf.Variable(tf.random_normal([64])),
    'bconv3': tf.Variable(tf.random_normal([64])),
    'bfc1': tf.Variable(tf.random_normal([1024])),
    'out': tf.Variable(tf.random_normal([n_class]))
}

# conv2d function for simplicity
def conv2d(x, W, b, strides=1):
    # Conv2D with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)

# maxpool2d function for simplicity
def maxpool2d(x, k=2):
    # MaxPool2D 
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')

# define CNN
def conv_net(x, weights, biases):
    
    # reshape input
    x = tf.reshape(x, shape=[-1, M, N, 1])

    # First Convolution Layer
    conv1 = conv2d(x, weights['wconv1'], biases['bconv1'])
    conv1 = maxpool2d(conv1, k=2)

    # Second Convolution Layer
    conv2 = conv2d(conv1, weights['wconv2'], biases['bconv2'])
    conv2 = maxpool2d(conv2, k=2)
    
    # Third Convolution Layer
    conv3 = conv2d(conv2, weights['wconv3'], biases['bconv3'])
    conv3 = maxpool2d(conv3, k=1)

    # Fully connected layer
    fc1 = tf.reshape(conv3, [-1, weights['wfc1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wfc1']), biases['bfc1'])
    fc1 = tf.nn.relu(fc1)

    # Class prediction
    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    return out
    
def nextBatch():
    idx = np.random.choice(train_size,size=batch_size,replace=False)
    batch_x = trainX[idx, :]
    batch_y = trainY[idx, :]
    return batch_x, batch_y  
    
# construct model
pred = conv_net(x, weights, biases)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32)) 

# Initializing the variables
init = tf.initialize_all_variables()

loss_vals = []

# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    step = 1
    while step * batch_size < iters:
        batch_x, batch_y = nextBatch()
        
        # Run optimization 
        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y} )
        if step % display_step == 0:
            # Calculate batch loss and accuracy
            loss, acc = sess.run([cost, accuracy], feed_dict={x: batch_x, y: batch_y})
            print("Iteration " + str(step*batch_size) + ", Minibatch Loss= " + \
                  "{:.6f}".format(loss) + ", Training Accuracy= " + \
                  "{:.5f}".format(acc))
        loss_vals.append(loss)
        step += 1
    print("Optimization Finished!")

    # Calculate accuracy for test images
    print("Testing Accuracy:", \
        sess.run(accuracy, feed_dict={x: testX, y: testY}))   
        
        
fig = plt.figure()
plt.plot(loss_vals)
plt.show()
    
    