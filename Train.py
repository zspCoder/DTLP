
import tensorflow as tf
import scipy.io as sio
import numpy as np
import os
import re
import Model
import h5py
from Utils import *

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


# Parameters
block_size  = 48        #Image block size
channel     = 31        #number of spectral bands
batch_size  = 64        #Batch size in training
phase       = 11        #Max phase number of HQS
epoch_num   = 400       #Max epoch number in training
learn_rate  = 0.0001    #Learning rate
rank        = 4         #CP rank

dataset   = 'Harvard'      #Dataset
continueTrain_flag = False #Retrain from the stored model if continueTrain_flag==True

# Date path
train_data_name = './Data/Train/Training_Data_%s_48.mat' % (dataset)
model_dir          = 'Model/%s_%dPhase_%dEpoch_%.5fLearnrate_%dRank/' % (dataset,phase,epoch_num,learn_rate,rank)
output_file_name   = 'Model/Log_%s_%dPhase_%dEpoch_%.5fLearnrate_%dRank.txt' % (dataset,phase,epoch_num,learn_rate,rank)
if not os.path.exists(model_dir):
        os.makedirs(model_dir)

# Load training data
print("...............................")
print("Load training data...")
Training_data   = h5py.File(train_data_name,'r')
Training_labels = Training_data['label']
Training_labels = np.transpose(Training_labels, (0, 3, 2, 1))
del Training_data
nrtrain         = Training_labels.shape[0]

# Define variables
gloabl_steps  = tf.Variable(0, trainable=False)
learning_rate = tf.train.exponential_decay(learning_rate=learn_rate, global_step=gloabl_steps, decay_steps=(nrtrain//batch_size)*10, decay_rate=0.95,
                                           staircase=True)
Cu       = tf.placeholder(tf.float32, [None, block_size, block_size, channel])
X_output = tf.placeholder(tf.float32, [None, block_size, block_size, channel])
b        = tf.zeros(shape=(tf.shape(X_output)[0], channel-1, tf.shape(X_output)[2], tf.shape(X_output)[3]))

# Forward imaging and Initialization
y  = Encode_CASSI(X_output,Cu)
x0 = Init_CASSI(y,Cu,channel)

# Model
Prediction = Model.Interface(x0, Cu, phase, rank, channel, reuse=False)
cost_all = tf.reduce_mean(tf.square(Prediction - X_output))
optm_all = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost_all, global_step=gloabl_steps)
init = tf.global_variables_initializer()

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
saver = tf.train.Saver(tf.global_variables(), max_to_keep=1000)
sess = tf.Session(config=config)
sess.run(init)


print("Training samples number: %d" % (nrtrain))
print("Phase number: %d" % (phase))
print("Image block size: %d" % (block_size))
print("Max epoch number: %d" % (epoch_num))
print("CP rank: %s" % (rank))
print("Dataset: %s" % (dataset))
print("...............................\n")

# Retrain from the stored model (if continueTrain_flag == True)
if continueTrain_flag:
    ckpt     = tf.train.get_checkpoint_state(model_dir)
    if ckpt and ckpt.model_checkpoint_path:
       saver.restore(sess, ckpt.model_checkpoint_path)
       ckpt_num_seq = re.findall(r"\d+\d*",ckpt.model_checkpoint_path)
    ckpt_num = int(ckpt_num_seq[-1])
else:
    ckpt_num = -1

#  Training
print('Initial epoch: %d' % (ckpt_num+1))
print("Strart Training...")
for epoch_i in range(ckpt_num+1, epoch_num):
    randidx_all = np.random.permutation(nrtrain)
    for batch_i in range(nrtrain // batch_size):
        randidx = randidx_all[batch_i*batch_size:(batch_i+1)*batch_size]
        batch_ys = Training_labels[randidx, :, :, :]
        Cu_input = np.zeros([block_size, block_size, channel])
        T = np.round(np.random.rand(block_size/2, block_size/2))
        T = np.concatenate([T,T],axis=0)
        T = np.concatenate([T,T],axis=1)
        for ch in range(channel):
            Cu_input[:,:,ch] = np.roll(T, shift=-ch, axis=0)
        Cu_input = np.expand_dims(Cu_input, axis=0)
        Cu_input = np.tile(Cu_input, [batch_size, 1, 1, 1])

        feed_dict = {X_output: batch_ys, Cu: Cu_input}
        sess.run(optm_all, feed_dict=feed_dict)
    	output_data = "[%03d/%03d/%03d] cost: %.6f  learningrate: %.6f \n" % (batch_i, nrtrain // batch_size, epoch_i, sess.run(cost_all, feed_dict=feed_dict), sess.run(learning_rate, feed_dict=feed_dict))
        print(output_data)

    output_file = open(output_file_name, 'a')
    output_file.write(output_data)
    output_file.close()

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    saver.save(sess, './%s/model_%d.cpkt' % (model_dir, epoch_i), write_meta_graph=False)
sess.close()
print("Training Finished")

