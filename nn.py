#!/usr/bin/env python3
import tensorflow as tf
import pickle
import numpy as np
import matplotlib.pyplot as plt

""" Helper functions for NN """
def generate_batch(data,labels,batch_size):
    idxList = np.random.randint(data.shape[0], size=batch_size)
    batch_data = []
    batch_labels = []
    for idx in idxList:
        batch_data.append(data[idx,:])
        batch_labels.append(labels[idx])
    return np.asarray(batch_data),np.asarray(batch_labels)

def load_data(fileName):
    with open(fileName, 'rb') as f:
        data = pickle.load(f)
    return np.asarray(data)



# Hyper-parameters
epochs = 500
batch_size = 256
learning_rate = 5*1e-3
lmb = 0.05
#Place holders for data
x = tf.placeholder(tf.float64,[None,7])
y = tf.placeholder(tf.float64,[None])
#Hidden Layer 1
Wh_1 = tf.Variable(tf.truncated_normal([7,16], mean=0.0, stddev=1.0/np.sqrt(7), dtype=tf.float64),name = 'Wh_1')
bh_1 = tf.Variable(tf.zeros(shape= [16],dtype=tf.float64),name = 'bh_1')
out1 = tf.nn.relu(tf.add(tf.matmul(x,Wh_1),bh_1))
#Hidden Layer 2
Wh_2 = tf.Variable(tf.truncated_normal([16,16], mean=0.0, stddev=1.0/np.sqrt(16), dtype=tf.float64),name = 'Wh_2')
bh_2 = tf.Variable(tf.zeros(shape= [16],dtype=tf.float64),name = 'bh_2')
out2 = tf.nn.relu(tf.add(tf.matmul(out1,Wh_2),bh_2))
#Hidden Layer 3
#Wh_3 = tf.Variable(tf.truncated_normal([3,2], mean=0.0, stddev=1.0/np.sqrt(3), dtype=tf.float64),name = 'Wh_3')
#bh_3 = tf.Variable(tf.zeros(shape= [2],dtype=tf.float64),name = 'bh_3')
#out3 = tf.nn.relu(tf.add(tf.matmul(out2,Wh_3),bh_3))
#Output
W_out = tf.Variable(tf.truncated_normal([16,1], mean=0.0, stddev=1.0/np.sqrt(16), dtype=tf.float64),name = 'W_out')
b_out = tf.Variable(tf.zeros(shape=[1],dtype=tf.float64),name = 'b_out')
out = tf.add(tf.matmul(out2,W_out),b_out)
#Regularization
regularizers = tf.nn.l2_loss(W_out) + tf.nn.l2_loss(Wh_1) + tf.nn.l2_loss(Wh_2) #+ tf.nn.l2_loss(Wh_3)
#MSE cost
cost = tf.reduce_mean(tf.square(tf.subtract(y,out)))
loss = tf.reduce_mean(cost + lmb*regularizers)
#Train step
global_step = tf.Variable(0, trainable=False)
#learning_rate = tf.train.exponential_decay(start_learning_rate, global_step,1000, 0.1, staircase=True)
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
init = tf.global_variables_initializer()
params = tf.trainable_variables()
print('Tunable parameters : {}'.format(len(params)))
gradients = tf.gradients(cost, params)
gradient_norm = tf.global_norm(gradients)
with tf.Session() as sess:
    train_loss = []
    val_loss = []
    steps = []
    norm = []
    sess.run(init)
    X_train = load_data('data_train.pkl')
    labels_train = load_data('labels_train.pkl')
    X_val = load_data('data_val.pkl')
    labels_val = load_data('labels_val.pkl')
    batches_per_epoch = X_train.shape[0]//batch_size
    print('Batches per epoch : {}'.format(batches_per_epoch))
    for step in range(batches_per_epoch*epochs):
        batch_data,batch_labels = generate_batch(data=X_train,labels=labels_train,batch_size = batch_size)
        sess.run([train_step],feed_dict={x:batch_data,y:np.divide(batch_labels,1000)})
        if step%10 == 0:
            steps.append(step)
            grad_norm = sess.run(gradient_norm,feed_dict={x:batch_data,y:np.divide(batch_labels,1000)})
            train_cost = sess.run(loss,feed_dict={x:batch_data,y:np.divide(batch_labels,1000)})
            val_cost = sess.run(loss,feed_dict={x:X_val,y:np.divide(labels_val,1000)})
            train_loss.append(train_cost)
            val_loss.append(val_cost)
            norm.append(grad_norm)
            print('Step {} :: Gradient norm : {}'.format(step,grad_norm))
            print('Step {} :: Training cost : {}'.format(step,train_cost))
            print('Step {} :: Validation cost: {}'.format(step,val_cost))
    #Plot
    plt.figure(1)
    plt.subplot(211)
    plt.plot(steps,train_loss,'r',label='Train Loss')
    plt.plot(steps,val_loss,'b',label='Val Loss')
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.legend(loc = 'upper left')
    plt.subplot(212)
    plt.plot(steps,norm,label='Gradient Norm')
    plt.xlabel('Steps')
    plt.ylabel('Gradient Norm')
    plt.legend(loc = 'upper left')
    plt.show()



