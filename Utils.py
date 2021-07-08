
import tensorflow as tf
import numpy as np
import math


def Encode_CASSI(x, Mask):
    y = tf.multiply(x, Mask)
    y = tf.reduce_sum(y, axis=3)
    return y

def Init_CASSI(y, Mask, channel):
    y1 = tf.expand_dims(y, axis=3)
    y2 = tf.tile(y1, [1, 1, 1, channel])
    x0 = tf.multiply(y2, Mask)
    return x0

def Cal_mse(im1, im2):
    return np.mean(np.square(im1 - im2), dtype=np.float64)


def Cal_PSNR(im_true, im_test):
    channel  = im_true.shape[2]
    im_true  = 255*im_true
    im_test  = 255*im_test

    psnr_sum = 0
    for i in range(channel):
        band_true = np.squeeze(im_true[:,:,i])
        band_test = np.squeeze(im_test[:,:,i])
        err       = Cal_mse(band_true, band_test)
        max_value = np.max(np.max(band_true))
        psnr_sum  = psnr_sum+10 * np.log10((max_value ** 2) / err)
    
    return psnr_sum/channel

def Fusion(patch_image, gt_image, patch_cassi, block_size, stride):
    height       = gt_image.shape[0]
    width        = gt_image.shape[1]
    channel      = gt_image.shape[2]
    result_image = np.zeros(gt_image.shape)
    weight_image = np.zeros(gt_image.shape)
    cassi_image  = np.zeros([height,width])
    
    len_cassi    = 542

    count = 0
    for x in range(0,height-channel+1-block_size+1,stride):
        for y in range(0,width-block_size+1,stride):
            result_image[x:x+block_size,y:y+block_size,:] = result_image[x:x+block_size,y:y+block_size,:]+patch_image[:,:,:,count]
            weight_image[x:x+block_size,y:y+block_size,:] = weight_image[x:x+block_size,y:y+block_size,:]+1
            cassi_image[x:x+block_size,y:y+block_size] = cassi_image[x:x+block_size,y:y+block_size]+np.squeeze(patch_cassi[count,:,:])
            count = count+1
    for ch in range(channel):
        result_image[:,:,ch] = np.roll(result_image[:,:,ch], shift=ch, axis=0)
        weight_image[:,:,ch] = np.roll(weight_image[:,:,ch], shift=ch, axis=0)
    moreRow      = int((math.floor((len_cassi-block_size+stride-1)/stride))*stride + block_size-len_cassi)
    result_image = result_image[int(30+math.floor(moreRow/2)):int(height-30-(moreRow-math.floor(moreRow/2))),8:width-8,:]
    gt_image     = gt_image[int(30+math.floor(moreRow/2)):int(height-30-(moreRow-math.floor(moreRow/2))),8:width-8,:]
    cassi_image  = cassi_image[int(math.floor(moreRow/2)):int(height-30-(moreRow-math.floor(moreRow/2))),8:width-8]
    weight_cassi = weight_image[int(math.floor(moreRow/2)):int(height-30-(moreRow-math.floor(moreRow/2))),8:width-8,0]
    weight_image = weight_image[int(30+math.floor(moreRow/2)):int(height-30-(moreRow-math.floor(moreRow/2))),8:width-8,:]
   
    result_image = result_image/weight_image
    cassi_image  = cassi_image/weight_cassi
    return result_image, gt_image, cassi_image