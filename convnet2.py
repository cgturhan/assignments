# -*- coding: utf-8 -*-
"""
Created on Fri Nov 11 17:22:08 2016

@author: ceren
"""

import tensorflow as tf
import numpy as np
import zipfile
from matplotlib import pyplot as plt

# Load dataset
zip_ref = zipfile.ZipFile("data/ORL_faces.npz.zip", 'r')
zip_ref.extractall("data/")
zip_ref.close()

data = np.load('data/ORL_faces.npz')

trainX = data['trainX']
testX = data['testX']
trainY = data['trainY']
testY = data['testY']

# Parameters
n_class = np.unique(trainY).size
n_input = trainX.shape[1]
train_size = trainX.shape[0]
test_size = testX.shape[0]
M, N = 112, 92
batch_size = 32
display_step = 10
learning_rate = .01
iters = 10000
dropout = .6

# Convert labels to one-hot vectors
trainY = np.eye(n_class)[trainY]
testY = np.eye(n_class)[testY]

# tf inputs, variables
x = tf.placeholder(tf.float32, [None, n_input])
y = tf.placeholder(tf.float32, [None, n_class])
dropout_prob = tf.placeholder(tf.float32)
weights = {
    # 3x3 conv, 1 input, 28 outputs
    'wconv1': tf.Variable(tf.random_normal([3, 3, 1, 28])),
    # 5x5 conv, 28 inputs, 32 outputs
    'wconv2': tf.Variable(tf.random_normal([5, 5, 28, 32])),
    # 7x7 conv, 32 inputs, 64 outputs
    'wconv3': tf.Variable(tf.random_normal([7, 7, 32, 64])),
    # fully connected, 7*6*64 inputs, 1024 outputs
    'wfc1': tf.Variable(tf.random_normal([7*6*64, 1024])),
    # fully connected, 1024 inputs, 1024 outputs
    'wfc2': tf.Variable(tf.random_normal([1024, 1024])),
    # 1024 inputs, 20 outputs   
    'out': tf.Variable(tf.random_normal([1024, n_class]))
}

biases = {
    'bconv1': tf.Variable(tf.random_normal([28])),
    'bconv2': tf.Variable(tf.random_normal([32])),
    'bconv3': tf.Variable(tf.random_normal([64])),
    'bfc1': tf.Variable(tf.random_normal([1024])),
    'bfc2': tf.Variable(tf.random_normal([1024])),
    'out': tf.Variable(tf.random_normal([n_class]))
}

# Conv2d function
def conv2d(x, W, b, strides=1):
    # Conv2D with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)

# Maxpool2d function
def maxpool2d(x, k=2):
    # MaxPool2D 
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')

# Define CNN
def conv_net(x, weights, biases, dropout):
    
    # reshape input
    x = tf.reshape(x, shape=[-1, M, N, 1])

    # First Convolution Layer
    conv1 = conv2d(x, weights['wconv1'], biases['bconv1'])
    conv1 = maxpool2d(conv1, k=2)

    # Second Convolution Layer
    conv2 = conv2d(conv1, weights['wconv2'], biases['bconv2'], strides = 2)
    conv2 = maxpool2d(conv2, k=2)
    
    # Third Convolution Layer
    conv3 = conv2d(conv2, weights['wconv3'], biases['bconv3'])
    conv3 = maxpool2d(conv3, k=2)

    # First Fully connected layer
    fc1 = tf.reshape(conv3, [-1, weights['wfc1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wfc1']), biases['bfc1'])
    fc1 = tf.nn.relu(fc1)
    
    # Dropout regularizaiton
    fc1 = tf.nn.dropout(fc1, dropout)
    
    # Second Fully connected layer
    fc2 = tf.add(tf.matmul(fc1, weights['wfc2']), biases['bfc2'])
    fc2 = tf.nn.relu(fc2)
    
    fc2 = tf.nn.dropout(fc2, dropout)

    # Class prediction
    out = tf.add(tf.matmul(fc2, weights['out']), biases['out'])
    return out
    
def nextBatch():
    idx = np.random.choice(train_size,size=batch_size,replace=False)
    batch_x = trainX[idx, :]
    batch_y = trainY[idx, :]
    return batch_x, batch_y  
    
def visualizationofWeights(weights):


    for wname, w in weights.items():
        # Take only conv weights
        if wname.startswith('wconv'): 
            
            plt.figure()

            nFilter = w.shape[3]
            k = w.shape[0]

            for i in range(nFilter):
                img = w[:,:,0,i]
                img = img.reshape(k,k)
                
                plt.subplot(1,nFilter,i)
                plt.cla()
                
                plt.imshow(img)
                
                frame = plt.gca()
                frame.axes.get_xaxis().set_ticks([])
                frame.axes.get_yaxis().set_ticks([])
                
            plt.title('Filters:' + wname[1:])
            plt.savefig('filters_'+ wname[1:] + '.png')
            

        
# Construct model
pred = conv_net(x, weights, biases, dropout_prob)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32)) 

# Initialize the variables
init = tf.initialize_all_variables()

losses = []

# Launch the graph
with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
    sess.run(init)
    step = 1
    while step * batch_size < iters:
        batch_x, batch_y = nextBatch()
        # Run optimization 
        _,w = sess.run([optimizer, weights], feed_dict={x: batch_x, y: batch_y, dropout_prob:dropout} )
            
        if step % display_step == 0:
            # Calculate batch loss and accuracy
            loss, acc = sess.run([cost, accuracy], feed_dict={x: batch_x, y: batch_y, dropout_prob: 1.0 })
            print("Iteration " + str(step*batch_size) + ", Minibatch Loss= " + \
                      "{:.6f}".format(loss) + ", Training Accuracy= " + \
                      "{:.5f}".format(acc))
            losses.append(loss)
        step += 1
    print("Optimization Finished!")
    visualizationofWeights(w)
    # Calculate accuracy for test images
    print("Testing Accuracy:", \
            sess.run(accuracy, feed_dict={x: testX, y: testY, dropout_prob: 1.0}))   
            
        
fig = plt.figure()
plt.plot(losses)
plt.show()
    

    
