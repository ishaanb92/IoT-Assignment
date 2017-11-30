import tensorflow as tf
import pickle
import numpy as np

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
epochs = 50
batch_size = 256
learning_rate = 1e-5
lmb = 0.5
#Place holders for data
x = tf.placeholder(tf.float64,[None,7])
y = tf.placeholder(tf.float64,[None])
#Hidden Layer 1
Wh_1 = tf.Variable(tf.truncated_normal([7,5], mean=0.0, stddev=1.0, dtype=tf.float64),name = 'Wh_1')
bh_1 = tf.Variable(tf.zeros(shape= [5],dtype=tf.float64),name = 'bh_1')
out1 = tf.nn.relu(tf.add(tf.matmul(x,Wh_1),bh_1))
#Hidden Layer 2
Wh_2 = tf.Variable(tf.truncated_normal([5,3], mean=0.0, stddev=1.0, dtype=tf.float64),name = 'Wh_1')
bh_2 = tf.Variable(tf.zeros(shape= [3],dtype=tf.float64),name = 'bh_1')
out2 = tf.nn.relu(tf.add(tf.matmul(out1,Wh_2),bh_2))
#Output
W_out = tf.Variable(tf.truncated_normal([3,1], mean=0.0, stddev=1.0, dtype=tf.float64),name = 'W_out')
b_out = tf.Variable(tf.zeros(shape=[1],dtype=tf.float64),name = 'b_out')
out = tf.add(tf.matmul(out2,W_out),b_out)
#Regularization
regularizers = tf.nn.l2_loss(W_out) + tf.nn.l2_loss(Wh_1) + tf.nn.l2_loss(Wh_2)
#MSE cost
cost = tf.reduce_mean(tf.square(tf.subtract(y,out)))
loss = tf.reduce_mean(cost + lmb*regularizers)
#Train step
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
init = tf.global_variables_initializer()
params = tf.trainable_variables()
print('Tunable parameters : {}'.format(len(params)))
gradients = tf.gradients(cost, params)
gradient_norm = tf.global_norm(gradients)
with tf.Session() as sess:
    sess.run(init)
    X_train = load_data('data_train.pkl')
    labels_train = load_data('labels_train.pkl')
    X_val = load_data('data_val.pkl')
    labels_val = load_data('labels_val.pkl')
    batches_per_epoch = X_train.shape[0]//batch_size
    print('Batches per epoch : {}'.format(batches_per_epoch))
    for ep in range(epochs):
        for step in range(batches_per_epoch):
            batch_data,batch_labels = generate_batch(data=X_train,labels=labels_train,batch_size = batch_size)
            sess.run([train_step],feed_dict={x:batch_data,y:batch_labels})
            if step%10 == 0:
                grad_norm = sess.run([gradient_norm],feed_dict={x:batch_data,y:batch_labels})
                print('Epoch {} Step {} :: Gradient norm : {}'.format(ep,step,grad_norm))
                print('Epoch {} Step {} :: Training cost : {}'.format(ep,step,sess.run([cost],feed_dict={x:batch_data,y:batch_labels})))
                print('Epoch {} Step {} :: Validation cost: {}'.format(ep,step,sess.run([cost],feed_dict={x:X_val,y:labels_val})))
                #print('Epoch {} Step {} :: Predicted : {} True {}'.format(ep,step,sess.run([out],feed_dict={x:batch_data,y:batch_labels}),batch_labels))


