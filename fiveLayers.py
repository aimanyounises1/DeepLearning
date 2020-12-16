import tensorflow as tf
from tensorflow import keras
import numpy as np
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
(x_train,y_train),(x_test,y_test) = cifar10.load_data()
x = tf.placeholder(tf.float32,[None,3072])
# labels
y_ = tf.placeholder(tf.float32,[None,10])
keep_prob = tf.placeholder(tf.float32)
x_train = x_train.reshape(50000, 3072)
x_test = x_test.reshape(10000, 3072)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
(hidden1_size,hidden2_size,hidden3_size,hidden4_size,hidden5_size) = (200,400,200,100,50)
# convert class vectors to binary class matrices
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)
x_train = x_train.reshape(50000, 3072)
W1 = tf.Variable(tf.zeros([3072,hidden1_size]))
b1 = tf.Variable(tf.zeros(hidden1_size))
z1 = tf.nn.elu(tf.matmul(x,W1)+b1)
#dropout1 = tf.nn.dropout(z1,keep_prob=keep_prob)
W2 = tf.Variable(tf.truncated_normal([hidden1_size,hidden2_size],stddev=0.1))
b2 = tf.Variable(tf.constant(0.1,shape=[hidden2_size]))
z2 = tf.nn.elu(tf.matmul(z1,W2)+b2)
dropout2 = tf.nn.dropout(z2,keep_prob=keep_prob)

W3 = tf.Variable(tf.truncated_normal([hidden2_size,hidden3_size],stddev=0.1))
b3 = tf.Variable(tf.constant(0.1,shape=[hidden3_size]))
z3 = tf.nn.elu(tf.matmul(dropout2,W3)+b3)
dropout3 = tf.nn.dropout(z3,keep_prob=keep_prob)

W4 = tf.Variable(tf.truncated_normal([hidden3_size,hidden4_size],stddev=0.1))
b4 = tf.Variable(tf.constant(0.1,shape=[hidden4_size]))
z4 = tf.nn.elu(tf.matmul(dropout3,W4)+b4)
dropout4 = tf.nn.dropout(z4,keep_prob=keep_prob)

W5 = tf.Variable(tf.truncated_normal([hidden4_size,hidden5_size],stddev=0.1))
b5 = tf.Variable(tf.constant(0.1,shape=[hidden5_size]))
z5 = tf.nn.elu(tf.matmul(dropout4,W5)+b5)
dropout5 = tf.nn.dropout(z5,keep_prob=keep_prob)

W6 = tf.Variable(tf.truncated_normal([hidden5_size,10],stddev=0.1))
b6 = tf.Variable(tf.constant(0.1,shape=[10]))
# predictions
y = tf.nn.softmax(tf.matmul(dropout5,W6)+b6)
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y),reduction_indices=[1]))
# let's train to get the minimum loss using back+forward propagation
train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y, 1),tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
sess = tf.Session()
sess.run(tf.global_variables_initializer())
for epoch in range(0,1000):
    print("Train set cost function =",epoch,sess.run([train_step,cross_entropy], feed_dict={x: x_train, y_: y_train,keep_prob:1}))  # BGD
    print("Train set accuracy =",epoch,sess.run(accuracy, feed_dict={x: x_train, y_: y_train,keep_prob:1}))
    if (epoch % 100 == 0):
        print("Test set cost func = ",epoch,sess.run([train_step, cross_entropy], feed_dict={x: x_train, y_: y_train,keep_prob:1}))  # BGD
        print("Test set accuracy =",epoch,sess.run(accuracy, feed_dict={x: x_test, y_: y_test,keep_prob:1}))
