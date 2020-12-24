#import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf
from tensorflow import keras
import numpy as np
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
# loading the data from keras.cifar10
(x_train,y_train),(x_test,y_test) = cifar10.load_data()
x = tf.placeholder(tf.float32,[None,3072])
# labels
y_ = tf.placeholder(tf.float32,[None,10])
keep_prob = tf.compat.v1.placeholder(tf.float32)
# reshaping the test data and the train
x_train = x_train.reshape(50000, 3072)
x_test = x_test.reshape(10000, 3072)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
# normalizing the data for train_x and train_y
x_train /= 255
x_test /= 255
(hidden1_size,hidden2_size,hidden3_size,hidden4_size,hidden5_size) = (512,256,128,64,32)
# convert class vectors to binary class matrices
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)
x_train = x_train.reshape(50000, 3072)
W1 = tf.Variable(tf.truncated_normal([3072,hidden1_size],stddev=0.1))
b1 = tf.Variable(tf.zeros(hidden1_size))
z1 = tf.nn.elu(tf.matmul(x,W1)+b1)
W2 = tf.Variable(tf.truncated_normal([hidden1_size,hidden2_size],stddev=0.1))
b2 = tf.Variable(tf.constant(0.1,shape=[hidden2_size]))
z2 = tf.nn.elu(tf.matmul(z1,W2)+b2)

W3 = tf.Variable(tf.truncated_normal([hidden2_size,hidden3_size],stddev=0.1))
b3 = tf.Variable(tf.constant(0.1,shape=[hidden3_size]))
z3 = tf.nn.elu(tf.matmul(z2,W3)+b3)#z3

W4 = tf.Variable(tf.truncated_normal([hidden3_size,hidden4_size],stddev=0.1))
b4 = tf.Variable(tf.constant(0.1,shape=[hidden4_size]))
z4 = tf.nn.elu(tf.matmul(z3,W4)+b4)

W5 = tf.Variable(tf.truncated_normal([hidden4_size,hidden5_size],stddev=0.1))
b5 = tf.Variable(tf.constant(0.1,shape=[hidden5_size]))
z5 = tf.nn.elu(tf.matmul(z4,W5)+b5)

W6 = tf.Variable(tf.truncated_normal([hidden5_size,10],stddev=0.1))
b6 = tf.Variable(tf.constant(0.1,shape=[10]))
# predictions
y = tf.nn.softmax(tf.matmul(z5,W6)+b6)
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y),reduction_indices=[1]))
# let's train to get the minimum loss using back+forward propagation
train_step = tf.train.GradientDescentOptimizer(0.0001).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y, 1),tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
sess = tf.Session()
sess.run(tf.global_variables_initializer())
for i in range(0,1001):
    #if (i % 100 == 0):
    print(i," Train set cost function =",sess.run([train_step,cross_entropy], feed_dict={x: x_train, y_: y_train})) # BGD
    TrainAcc = sess.run(accuracy,feed_dict={x:x_train,y_:y_train})
    print(TrainAcc)
    if (i % 100 == 0):
        print(i," Test Cost Function" ,sess.run([train_step,cross_entropy],feed_dict={x:x_test,y_:y_test}))
        print(i," Test set accuracy =",sess.run(accuracy, feed_dict={x: x_test, y_: y_test}))