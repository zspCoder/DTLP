
import tensorflow as tf
import numpy as np


def def_con2d_weight(w_shape, w_name):
    # Define the net weights
    weights = tf.get_variable(shape=w_shape, initializer=tf.contrib.layers.xavier_initializer_conv2d(), name='Weights_%s' % w_name)
    return weights

def ResBlock(input, order, filter_num):
    # Residual block for DRTLM
	xup   = RTGB(input, 'up_order_%d' % (order), filter_num)
	x_res = input - xup
	xdn   = RTGB(x_res, 'down_order_%d' % (order), filter_num)
	xdn   = xdn + x_res
	return xup, xdn

def RTGB(input, order_name, filter_num=64):
    # Rank-1 tensor generating block
	gap_Height   =  tf.reduce_mean(tf.reduce_mean(input, axis=2, keepdims=True), axis=3, keepdims=True)
	gap_Weight   =  tf.reduce_mean(tf.reduce_mean(input, axis=1, keepdims=True), axis=3, keepdims=True)
	gap_Channel  =  tf.reduce_mean(tf.reduce_mean(input, axis=1, keepdims=True), axis=2, keepdims=True)

	weights_H = def_con2d_weight([1, 1, 1, 1], 'cp_con1d_hconv_%s' % (order_name))
	weights_W = def_con2d_weight([1, 1, 1, 1], 'cp_con1d_wconv_%s' % (order_name))
	weights_C = def_con2d_weight([1, 1, filter_num, filter_num], 'cp_con1d_cconv_%s' % (order_name))

	convHeight_GAP    = tf.nn.sigmoid(tf.nn.conv2d(gap_Height, weights_H, strides=[1, 1, 1, 1], padding='SAME'), name = 'sig_hgap_%s' % (order_name));
	convWeight_GAP    = tf.nn.sigmoid(tf.nn.conv2d(gap_Weight, weights_W, strides=[1, 1, 1, 1], padding='SAME'), name = 'sig_wgap_%s' % (order_name));
	convChannel_GAP   = tf.nn.sigmoid(tf.nn.conv2d(gap_Channel, weights_C, strides=[1, 1, 1, 1], padding='SAME'), name = 'sig_cgap_%s' % (order_name));

	vecConHeight_GAP  = tf.reshape(convHeight_GAP, [tf.shape(convHeight_GAP)[0], tf.shape(convHeight_GAP)[1],1])
	vecConWeight_GAP  = tf.reshape(convWeight_GAP, [tf.shape(convWeight_GAP)[0], 1, tf.shape(convWeight_GAP)[2]])
	vecConChannel_GAP = tf.reshape(convChannel_GAP, [tf.shape(convChannel_GAP)[0], 1, tf.shape(convChannel_GAP)[3]])

	matHWmulT    = tf.matmul(vecConHeight_GAP, vecConWeight_GAP)
	vecHWmulT    = tf.reshape(matHWmulT, [tf.shape(matHWmulT)[0], tf.shape(matHWmulT)[1] * tf.shape(matHWmulT)[2], 1])
	matHWCmulT   = tf.matmul(vecHWmulT, vecConChannel_GAP)
	recon        = tf.reshape(matHWCmulT, [tf.shape(input)[0], tf.shape(input)[1], tf.shape(input)[2], tf.shape(input)[3]])
	return recon

def DRTLM(input, rank, filter_num):
    #  Discriminative rank-1 tensor learning module
	(xup, xdn) = ResBlock(input, 0, filter_num)
	temp_xup   = xdn
	output     = xup
	for i in range(1,rank):
		(temp_xup,temp_xdn) = ResBlock(temp_xup, i, filter_num)
		xup      = xup + temp_xup       
		output   = tf.concat([output, xup],3)
		temp_xup = temp_xdn
	return output

def Encoding(input, filter_size, filter_num):
    # Get deep feature maps
    weights_pro_0 = def_con2d_weight([filter_size, filter_size, filter_num, filter_num], 'fproject_con2d_conv_0')
    input_temp    = tf.nn.relu(tf.nn.conv2d(input, weights_pro_0, strides=[1, 1, 1, 1], padding='SAME'))

    weights_pro_1 = def_con2d_weight([filter_size, filter_size, filter_num, filter_num], 'fproject_con2d_conv_1')
    output        = tf.nn.conv2d(input_temp, weights_pro_1, strides=[1, 1, 1, 1], padding='SAME')
    return output

def Fusion(input, xt, filter_size, filter_num, channel_num):
    # Aggregate multiple rank-1 tensors into a low-rank tensor
    weights_attention = def_con2d_weight([filter_size, filter_size, filter_num, channel_num], 'IRecon_attention_con2d_conv')
    attention_map     = tf.nn.conv2d(input, weights_attention, strides=[1, 1, 1, 1], padding='SAME')
    output            = tf.multiply(xt,attention_map)
    return output

def Recon(xt, x0, Cu, layer_no, channel = 31, rank = 4):
    # Parameters
    deta        = tf.Variable(0.04, dtype=tf.float32, name='deta_%d' % layer_no)
    eta         = tf.Variable(0.8, dtype=tf.float32, name='eta_%d' % layer_no)
    filter_size = 3
    filter_num  = 64

    weights_main_0 = def_con2d_weight([filter_size, filter_size, channel, filter_num], 'main_con2d_conv_0')
    weights_main_1 = def_con2d_weight([filter_size, filter_size,  filter_num, channel], 'main_con2d_conv_1')
    
    # Low-rank Tensor Recovery
    x_feature_0        = tf.nn.conv2d(xt, weights_main_0, strides=[1, 1, 1, 1], padding='SAME')
    x_feature_1        = Encoding(x_feature_0, filter_size, filter_num)
    attention_map_cat  = DRTLM(x_feature_1, rank, filter_num)

    x_feature_lowrank  = Fusion(attention_map_cat, x_feature_1, filter_size, filter_num * rank, filter_num)
    x_mix              = x_feature_lowrank + x_feature_0
    z  = tf.nn.relu(tf.nn.conv2d(x_mix, weights_main_1, strides=[1, 1, 1, 1], padding='SAME'))

    # Linear Projection
    yt  = tf.multiply(xt, Cu)
    yt  = tf.reduce_sum(yt, axis=3)
    yt1 = tf.expand_dims(yt, axis=3)
    yt2 = tf.tile(yt1, [1, 1, 1, channel])
    xt2 = tf.multiply(yt2, Cu)  # PhiT*Phi*xt
    x   = tf.scalar_mul(1-deta*eta, xt) - tf.scalar_mul(deta, xt2) + tf.scalar_mul(deta, x0) + tf.scalar_mul(deta*eta, z)
    return x


def Interface(x, Cu, phase, rank, channel, reuse):
    '''
    Input parameters:
    x-----Initialized image
    Cu----CASSI mask
    phase-----Max phase number
    rank---CP rank
    channel---Spectral band number

    Output parameters:
    xt----Reconstructed image
    '''

    xt = x
    for i in range(phase):
        with tf.variable_scope('Phase_%d' %i, reuse=reuse):
            xt = Recon(xt, x, Cu, i, channel, rank)
    return xt
