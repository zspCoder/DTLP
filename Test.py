
import tensorflow as tf
import scipy.io as sio
import numpy as np
from time import time
import os
import Model
from Utils import *


os.environ["CUDA_DEVICE_ORDER"]    = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"       

# Parameters
cpkt_model_number = 271 #Checkpoint

phase             = 11
block_size        = 48
channel           = 31
rank              = 4
learn_rate        = 0.0001
epoch_num         = 400


dataset           = 'ICVL'

# Image size
'''
We provide images with full size 512 * 512 * 31 here.
Note that, for the comparison with previous methods, 
we evaluate the results of the central area with 256 * 256 * 31 in the paper.
'''
height            = 512
width             = 512
stride            = 24


model_dir         = './Model/%s_%dPhase_%dEpoch_%.5fLearnrate_%dRank' % (dataset,phase,epoch_num,learn_rate,rank)
result_dir        = 'Result/%s_%dCkpt_%dPhase_%dEpoch_%.5fLearnrate_%dRank' % (dataset,cpkt_model_number,phase,epoch_num,learn_rate,rank)
test_data_dir     = './Data/Test/%s%d' % (dataset, block_size)
if not os.path.exists(result_dir):
        os.makedirs(result_dir)

Cu = tf.placeholder(tf.float32, [None, block_size, block_size, channel])
X_output = tf.placeholder(tf.float32, [None, block_size, block_size, channel])
b = tf.zeros(shape=(tf.shape(X_output)[0], channel-1, tf.shape(X_output)[2], tf.shape(X_output)[3]))


y  = Encode_CASSI(X_output,Cu)
x0 = Init_CASSI(y,Cu,channel)
Prediction = Model.Interface(x0, Cu, phase, rank, channel, reuse=False)

# Model
cost_all = tf.reduce_mean(tf.square(Prediction - X_output))
optm_all = tf.train.AdamOptimizer(learning_rate=learn_rate).minimize(cost_all)
init     = tf.global_variables_initializer()
config   = tf.ConfigProto()
config.gpu_options.allow_growth = True
saver    = tf.train.Saver(tf.global_variables(), max_to_keep=100)
sess     = tf.Session(config=config)

saver.restore(sess, './%s/model_%d.cpkt' % (model_dir, cpkt_model_number))

filepaths = os.listdir(test_data_dir)
ImgNum    = len(filepaths)

if dataset=='Harvard':
    batch = 77
elif dataset=='ICVL':
    batch = 77

Cu_input = np.zeros([block_size, block_size, channel])
T  = np.round(np.random.rand(block_size/2, block_size/2))
T  = np.concatenate([T,T],axis=0)
T  = np.concatenate([T,T],axis=1)
for ch in range(channel):
        Cu_input[:,:,ch] = np.roll(T, shift=-ch, axis=0)
Cu_input = np.expand_dims(Cu_input, axis=0)
Cu_input = np.tile(Cu_input, [batch, 1, 1, 1])

print("\n...............................")
print('Dataset: %s'%(dataset))
print('Resolution: %d * %d * %d'%(height,width,channel))
print('Total number of images: %d' % (ImgNum))
print("...............................\n")

imgCnt   = 1
time_sum = 0
psnr_sum = 0
for img_no in range(ImgNum):

    imgName  = filepaths[img_no]
    imgName  = imgName[0:-4]
    testData = sio.loadmat(test_data_dir+'/'+filepaths[img_no])
        
    gt_image = testData['hyper_image']
    patch_image = testData['patch_image']
    patch_image = np.transpose(patch_image, (3, 0, 1, 2))
    
    print('Reconstructing image #%d' % (imgCnt))
    patchNum = patch_image.shape[0]
    for i in range(patchNum // batch):
        start = time()
        xoutput = patch_image[i * batch:(i + 1) * batch]

        Prediction_value = sess.run(Prediction, feed_dict={X_output: xoutput, Cu: Cu_input})
        end = time()

        y_value = sess.run(y, feed_dict={X_output: xoutput, Cu: Cu_input})
        cost_all_value = sess.run(cost_all, feed_dict={X_output: xoutput, Cu: Cu_input})
        print("Batch %d, run time: %.4f, loss sym: %.4f" % (i, (end - start), cost_all_value))
        if imgCnt>1:
            time_sum = time_sum + end - start
        
        Prediction_patch = np.transpose(Prediction_value, (1, 2, 3, 0))
        if i == 0:
            output = Prediction_patch
            cassi  = y_value
        else:
            output = np.concatenate([output, Prediction_patch], axis=3)
            cassi  = np.concatenate([cassi, y_value], axis=0)
        #print(output.shape)

 
    # Aggregate the oblique blocks into HSI
    result_image, gt_image, cassi_image = Fusion(output, gt_image, cassi, block_size, stride)
    
    psnr     = Cal_PSNR(gt_image, result_image)
    psnr_sum = psnr_sum + psnr
    
    print('Reconstructed PSNR: %.3f' % (psnr))
    print("...............................\n")
    
    out_dict = {'rec_image': result_image,
                'cassi_image':cassi_image,
                'gt_image': gt_image}
    out_filename = '%s/%s.mat' % (result_dir, imgName)
    sio.savemat(out_filename, out_dict)
    imgCnt = imgCnt + 1

sess.close()

print('Check point: %d'%(cpkt_model_number))
print('Average PSNR: %.3f dB'%(psnr_sum/ImgNum))
print('Average consuming time: %.3f secs'%(time_sum/(ImgNum-1)))

print("Reconstruction Finished")
