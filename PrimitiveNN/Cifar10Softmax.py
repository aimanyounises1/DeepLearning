import tensorflow as tf
from tensorflow import keras
import numpy as np
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
(x_train,y_train),(x_test,y_test) = cifar10.load_data()
x = tf.placeholder(tf.float32,[None,3072])
# labels
y_ = tf.placeholder(tf.float32,[None,10])
x_train = x_train.reshape(50000, 3072)
x_test = x_test.reshape(10000, 3072)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
# convert class vectors to binary class matrices
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)
x_train = x_train.reshape(50000, 3072)
W1 = tf.Variable(tf.zeros([3072,10]))
b1 = tf.Variable(tf.zeros([10]))
y = tf.nn.softmax(tf.matmul(x,W1)+b1)
# predictions
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y),reduction_indices=[1]))
# let's train to get the minimum loss using back+forward propagation
train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y, 1),tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
sess = tf.Session()
sess.run(tf.global_variables_initializer())
for i in range(0,1001):
    print("Train set cost function =", i,
          sess.run([train_step, cross_entropy], feed_dict={x: x_train, y_: y_train}))  # BGD
    TrainAcc = sess.run(accuracy, feed_dict={x: x_train, y_: y_train})
    print("Train accuracy = ",TrainAcc)
    if (i % 100 == 0):
        print(i ," Test Cost Function",  sess.run([train_step, cross_entropy], feed_dict={x: x_test, y_: y_test}))
        print(i," Test set accuracy =", i, sess.run(accuracy, feed_dict={x: x_test, y_: y_test}))