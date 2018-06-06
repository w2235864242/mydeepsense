from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import numpy as np
from datetime import datetime
import os.path
import time

import tensorflow.python.platform
from tensorflow.python.platform import gfile
#import plot

import time
import math
import os
import sys
FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_dir', 'test_train/',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_string('log_dir', 'logger/',
                           """Directory where to write event logs """
                           """and checkpoint.""")

layers = tf.contrib.layers

SEPCTURAL_SAMPLES = 10
FEATURE_DIM = SEPCTURAL_SAMPLES*10*2
mag_dim = 2*3*SEPCTURAL_SAMPLES
gps_dim = 2*5*SEPCTURAL_SAMPLES
light_dim = 2*1*SEPCTURAL_SAMPLES
pressure_dim = 2*1*SEPCTURAL_SAMPLES
CONV_LEN = 3
CONV_LEN_INTE = 3#4
CONV_LEN_LAST = 3#5
CONV_NUM = 64
CONV_MERGE_LEN = 8
CONV_MERGE_LEN2 = 6
CONV_MERGE_LEN3 = 4
CONV_NUM2 = 64
INTER_DIM = 200
OUT_DIM = 3#len(idDict)
WIDE = 20
CONV_KEEP_PROB = 0.8

BATCH_SIZE = 128
TOTAL_ITER_NUM = 8000

SUMMARY_DIR = ""

select = 't'

metaDict = {'t':[27920, 340], 'b':[116870, 1413], 'c':[116020, 1477]}
TRAIN_SIZE = metaDict[select][0]
EVAL_DATA_SIZE = metaDict[select][1]
EVAL_ITER_NUM = int(math.ceil(EVAL_DATA_SIZE / BATCH_SIZE))
INITIAL_LEARNING_RATE = 0.01

###### Import training data
def read_audio_csv(filename_queue):
	reader = tf.TextLineReader()
	key, value = reader.read(filename_queue)
	defaultVal = [[0.] for idx in range(WIDE*FEATURE_DIM + 1)]#20*5s data

	fileData = tf.decode_csv(value, record_defaults=defaultVal)
	features = fileData[:WIDE*FEATURE_DIM]
	features = tf.reshape(features, [WIDE, FEATURE_DIM])   # 20*140
	labels = fileData[WIDE*FEATURE_DIM:]
	labels = tf.cast(labels[0], tf.int32)
	return features, labels

def input_pipeline(filenames, batch_size, shuffle_sample=True, num_epochs=None):
	filename_queue = tf.train.string_input_producer(filenames, shuffle=shuffle_sample)
	# filename_queue = tf.train.string_input_producer(filenames, num_epochs=TOTAL_ITER_NUM*EVAL_ITER_NUM*10000000, shuffle=shuffle_sample)
	example, label = read_audio_csv(filename_queue)
	min_after_dequeue = 1600#int(0.4*len(csvFileList)) #1000
	capacity = min_after_dequeue + 3 * batch_size
	if shuffle_sample:
		example_batch, label_batch = tf.train.shuffle_batch(
			[example, label], batch_size=batch_size, num_threads=16, capacity=capacity,
			min_after_dequeue=min_after_dequeue)
	else:
		example_batch, label_batch = tf.train.batch(
			[example, label], batch_size=batch_size, num_threads=16)
	return example_batch, label_batch

######

# def batch_norm_layer(inputs, phase_train, scope=None):
# 	return tf.cond(phase_train,
# 		lambda: layers.batch_norm(inputs, is_training=True, scale=True,
# 			updates_collections=None, scope=scope),
# 		lambda: layers.batch_norm(inputs, is_training=False, scale=True,
# 			updates_collections=None, scope=scope, reuse = True))
def loss_cross(logits, labels):
	sparse_labels = tf.reshape(labels, [BATCH_SIZE, 1])
	# indices = tf.reshape(tfrange(FLAGS.batch_size), [FLAGS.batch_size, 1])
	indices = tf.reshape(range(BATCH_SIZE), [BATCH_SIZE, 1])
	concated = tf.concat([indices, sparse_labels], 1)
	dense_labels = tf.sparse_to_dense(concated,
									  [BATCH_SIZE, OUT_DIM],
									  1.0, 0.0)
		# Calculate the average cross entropy loss across the batch.
	cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
		logits=logits, labels=dense_labels)
	loss = tf.reduce_mean(cross_entropy)
	return loss

def batch_norm_layer(inputs, phase_train, scope=None):
	if phase_train:
		return layers.batch_norm(inputs, is_training=True, scale=True,
			updates_collections=None, scope=scope)
	else:
		return layers.batch_norm(inputs, is_training=False, scale=True,
			updates_collections=None, scope=scope, reuse = True)
# def variable_summaries(var):
# 	with tf.name_scope('summaries'):
# 		mean = tf.reduce_mean(var)
# 		tf.summary.scalar('mean', mean)
# 		stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
# 		tf.summary.scalar('stddev', stddev)
# 		tf.summary.scalar('max', tf.reduce_max(var))
# 		tf.summary.scalar('min', tf.reduce_min(var))
# 		tf.summary.histogram('histogram', var)

def deepSense(inputs, train, reuse=False, name='deepSense'):
	with tf.variable_scope(name, reuse=reuse):
		used = tf.sign(tf.reduce_max(tf.abs(inputs), reduction_indices=2)) #(BATCH_SIZE, WIDE)
		length = tf.reduce_sum(used, reduction_indices=1) #(BATCH_SIZE)
		length = tf.cast(length, tf.int64)

		mask = tf.sign(tf.reduce_max(tf.abs(inputs), reduction_indices=2, keep_dims=True))
		mask = tf.tile(mask, [1,1,INTER_DIM]) # (BATCH_SIZE, WIDE, INTER_DIM)
		avgNum = tf.reduce_sum(mask, reduction_indices=1) #(BATCH_SIZE, INTER_DIM)
		with tf.variable_scope("input"):
			# inputs shape (BATCH_SIZE, WIDE, FEATURE_DIM)
			sensor_inputs = tf.expand_dims(inputs, axis=3)  # di sanwei zengjia yi wei
			# sensor_inputs shape (BATCH_SIZE, WIDE, FEATURE_DIM, CHANNEL=1)
			# acc_inputs, gyro_inputs = tf.split(sensor_inputs, num_or_size_splits=2, axis=2)

			mag_inputs = tf.slice(sensor_inputs,[0,0,0,0],[BATCH_SIZE,WIDE,mag_dim,1])  #tf.slice(input_, begin, size, name=None)
			gps_inputs = tf.slice(sensor_inputs,[0,0,mag_dim,0],[BATCH_SIZE,WIDE,gps_dim,1])
			light_inputs = tf.slice(sensor_inputs,[0,0,mag_dim+gps_dim,0],[BATCH_SIZE,WIDE,light_dim,1])
			pressure_inputs = tf.slice(sensor_inputs,[0,0,mag_dim+gps_dim+light_dim,0],[BATCH_SIZE,WIDE,pressure_dim,1])
		with tf.variable_scope("magnetic"):
			mag_conv1 = layers.convolution2d(mag_inputs, CONV_NUM, kernel_size=[1, 2*3*CONV_LEN],
							stride=[1, 2*3], padding='VALID', activation_fn=None, data_format='NHWC', scope='mag_conv1')
			mag_conv1 = batch_norm_layer(mag_conv1, train, scope='mag_BN1')
			mag_conv1 = tf.nn.relu(mag_conv1)
			mag_conv1_shape = mag_conv1.get_shape().as_list()
			mag_conv1 = layers.dropout(mag_conv1, CONV_KEEP_PROB, is_training=train,
				noise_shape=[mag_conv1_shape[0], 1, 1, mag_conv1_shape[3]], scope='mag_dropout1')

			mag_conv2 = layers.convolution2d(mag_conv1, CONV_NUM, kernel_size=[1, CONV_LEN_INTE],
							stride=[1, 1], padding='VALID', activation_fn=None, data_format='NHWC', scope='mag_conv2')
			mag_conv2 = batch_norm_layer(mag_conv2, train, scope='mag_BN2')
			mag_conv2 = tf.nn.relu(mag_conv2)
			mag_conv2_shape = mag_conv2.get_shape().as_list()
			mag_conv2 = layers.dropout(mag_conv2, CONV_KEEP_PROB, is_training=train,
				noise_shape=[mag_conv2_shape[0], 1, 1, mag_conv2_shape[3]], scope='mag_dropout2')

			mag_conv3 = layers.convolution2d(mag_conv2, CONV_NUM, kernel_size=[1, CONV_LEN_LAST],
							stride=[1, 1], padding='VALID', activation_fn=None, data_format='NHWC', scope='mag_conv3')
			mag_conv3 = batch_norm_layer(mag_conv3, train, scope='mag_BN3')
			mag_conv3 = tf.nn.relu(mag_conv3)
			mag_conv3_shape = mag_conv3.get_shape().as_list()
			mag_conv_out = tf.reshape(mag_conv3, [mag_conv3_shape[0], mag_conv3_shape[1], mag_conv3_shape[2], 1, mag_conv3_shape[3]])

		with tf.variable_scope("GPS"):
			gps_conv1 = layers.convolution2d(gps_inputs, CONV_NUM, kernel_size=[1, 2*5*CONV_LEN],
							stride=[1, 2*5], padding='VALID', activation_fn=None, data_format='NHWC', scope='gps_conv1')
			gps_conv1 = batch_norm_layer(gps_conv1, train, scope='gps_BN1')
			gps_conv1 = tf.nn.relu(gps_conv1)
			gps_conv1_shape = gps_conv1.get_shape().as_list()
			gps_conv1 = layers.dropout(gps_conv1, CONV_KEEP_PROB, is_training=train,
				noise_shape=[gps_conv1_shape[0], 1, 1, gps_conv1_shape[3]], scope='gps_dropout1')

			gps_conv2 = layers.convolution2d(gps_conv1, CONV_NUM, kernel_size=[1, CONV_LEN_INTE],
							stride=[1, 1], padding='VALID', activation_fn=None, data_format='NHWC', scope='gps_conv2')
			gps_conv2 = batch_norm_layer(gps_conv2, train, scope='gps_BN2')
			gps_conv2 = tf.nn.relu(gps_conv2)
			gps_conv2_shape = gps_conv2.get_shape().as_list()
			gps_conv2 = layers.dropout(gps_conv2, CONV_KEEP_PROB, is_training=train,
				noise_shape=[gps_conv2_shape[0], 1, 1, gps_conv2_shape[3]], scope='gps_dropout2')

			gps_conv3 = layers.convolution2d(gps_conv2, CONV_NUM, activation_fn=None, kernel_size=[1, CONV_LEN_LAST],
							stride=[1, 1], padding='VALID', data_format='NHWC', scope='gps_conv3')
			gps_conv3 = batch_norm_layer(gps_conv3, train, scope='gps_BN3')
			gps_conv3 = tf.nn.relu(gps_conv3)
			gps_conv3_shape = gps_conv3.get_shape().as_list()
			gps_conv_out = tf.reshape(gps_conv3, [gps_conv3_shape[0], gps_conv3_shape[1], gps_conv3_shape[2], 1, gps_conv3_shape[3]])
		with tf.variable_scope("light"):
			light_conv1 = layers.convolution2d(light_inputs, CONV_NUM, kernel_size=[1, 2 * 1 * CONV_LEN],
											 stride=[1, 2 * 1], padding='VALID', activation_fn=None, data_format='NHWC',
											 scope='light_conv1')
			light_conv1 = batch_norm_layer(light_conv1, train, scope='light_BN1')
			light_conv1 = tf.nn.relu(light_conv1)
			light_conv1_shape = light_conv1.get_shape().as_list()
			light_conv1 = layers.dropout(light_conv1, CONV_KEEP_PROB, is_training=train,
									   noise_shape=[light_conv1_shape[0], 1, 1, light_conv1_shape[3]], scope='light_dropout1')

			light_conv2 = layers.convolution2d(light_conv1, CONV_NUM, kernel_size=[1, CONV_LEN_INTE],
											 stride=[1, 1], padding='VALID', activation_fn=None, data_format='NHWC',
											 scope='light_conv2')
			light_conv2 = batch_norm_layer(light_conv2, train, scope='light_BN2')
			light_conv2 = tf.nn.relu(light_conv2)
			light_conv2_shape = light_conv2.get_shape().as_list()
			light_conv2 = layers.dropout(light_conv2, CONV_KEEP_PROB, is_training=train,
									   noise_shape=[light_conv2_shape[0], 1, 1, light_conv2_shape[3]], scope='light_dropout2')

			light_conv3 = layers.convolution2d(light_conv2, CONV_NUM, activation_fn=None, kernel_size=[1, CONV_LEN_LAST],
											 stride=[1, 1], padding='VALID', data_format='NHWC', scope='light_conv3')
			light_conv3 = batch_norm_layer(light_conv3, train, scope='light_BN3')
			light_conv3 = tf.nn.relu(light_conv3)
			light_conv3_shape = light_conv3.get_shape().as_list()
			light_conv_out = tf.reshape(light_conv3,
									  [light_conv3_shape[0], light_conv3_shape[1], light_conv3_shape[2], 1, light_conv3_shape[3]])

		with tf.variable_scope("pressure"):
			pressure_conv1 = layers.convolution2d(pressure_inputs, CONV_NUM, kernel_size=[1, 2 * 1 * CONV_LEN],
											 stride=[1, 2 * 1], padding='VALID', activation_fn=None, data_format='NHWC',
											 scope='pressure_conv1')
			pressure_conv1 = batch_norm_layer(pressure_conv1, train, scope='pressure_BN1')
			pressure_conv1 = tf.nn.relu(pressure_conv1)
			pressure_conv1_shape = pressure_conv1.get_shape().as_list()
			pressure_conv1 = layers.dropout(pressure_conv1, CONV_KEEP_PROB, is_training=train,
									   noise_shape=[pressure_conv1_shape[0], 1, 1, pressure_conv1_shape[3]], scope='pressure_dropout1')

			pressure_conv2 = layers.convolution2d(pressure_conv1, CONV_NUM, kernel_size=[1, CONV_LEN_INTE],
											 stride=[1, 1], padding='VALID', activation_fn=None, data_format='NHWC',
											 scope='pressure_conv2')
			pressure_conv2 = batch_norm_layer(pressure_conv2, train, scope='pressure_BN2')
			pressure_conv2 = tf.nn.relu(pressure_conv2)
			pressure_conv2_shape = pressure_conv2.get_shape().as_list()
			pressure_conv2 = layers.dropout(pressure_conv2, CONV_KEEP_PROB, is_training=train,
									   noise_shape=[pressure_conv2_shape[0], 1, 1, pressure_conv2_shape[3]], scope='pressure_dropout2')

			pressure_conv3 = layers.convolution2d(pressure_conv2, CONV_NUM, activation_fn=None, kernel_size=[1, CONV_LEN_LAST],
											 stride=[1, 1], padding='VALID', data_format='NHWC', scope='pressure_conv3')
			pressure_conv3 = batch_norm_layer(pressure_conv3, train, scope='pressure_BN3')
			pressure_conv3 = tf.nn.relu(pressure_conv3)
			pressure_conv3_shape = pressure_conv3.get_shape().as_list()
			pressure_conv_out = tf.reshape(pressure_conv3,
									  [pressure_conv3_shape[0], pressure_conv3_shape[1], pressure_conv3_shape[2], 1, pressure_conv3_shape[3]])

		with tf.variable_scope("combine_sensor"):
			sensor_conv_in = tf.concat([mag_conv_out, gps_conv_out, light_conv_out, pressure_conv_out], 3)
			senor_conv_shape = sensor_conv_in.get_shape().as_list()
			sensor_conv_in = layers.dropout(sensor_conv_in, CONV_KEEP_PROB, is_training=train,
				noise_shape=[senor_conv_shape[0], 1, 1, 1, senor_conv_shape[4]], scope='sensor_dropout_in')
			sensor_conv_in = tf.reshape(sensor_conv_in, [senor_conv_shape[0], senor_conv_shape[1], mag_conv3_shape[2]+gps_conv3_shape[2]+light_conv3_shape[2]+pressure_conv3_shape[2],  pressure_conv3_shape[3]])

			sensor_conv1 = layers.convolution2d(sensor_conv_in, CONV_NUM2, kernel_size=[1, 2*CONV_MERGE_LEN],
							stride=[1, 2], padding='SAME', activation_fn=None, data_format='NHWC', scope='sensor_conv1')
			sensor_conv1 = batch_norm_layer(sensor_conv1, train, scope='sensor_BN1')
			sensor_conv1 = tf.nn.relu(sensor_conv1)
			sensor_conv1_shape = sensor_conv1.get_shape().as_list()
			sensor_conv1 = layers.dropout(sensor_conv1, CONV_KEEP_PROB, is_training=train,
				noise_shape=[sensor_conv1_shape[0], 1, 1, sensor_conv1_shape[3]], scope='sensor_dropout1')

			sensor_conv2 = layers.convolution2d(sensor_conv1, CONV_NUM2, kernel_size=[1, 2*CONV_MERGE_LEN2],
							stride=[1, 2], padding='SAME', activation_fn=None, data_format='NHWC', scope='sensor_conv2')
			sensor_conv2 = batch_norm_layer(sensor_conv2, train, scope='sensor_BN2')
			sensor_conv2 = tf.nn.relu(sensor_conv2)
			sensor_conv2_shape = sensor_conv2.get_shape().as_list()
			sensor_conv2 = layers.dropout(sensor_conv2, CONV_KEEP_PROB, is_training=train,
				noise_shape=[sensor_conv2_shape[0], 1, 1, sensor_conv2_shape[3]], scope='sensor_dropout2')

			sensor_conv3 = layers.convolution2d(sensor_conv2, CONV_NUM2, kernel_size=[1, 2*CONV_MERGE_LEN3],
							stride=[1, 2], padding='SAME', activation_fn=None, data_format='NHWC', scope='sensor_conv3')
			sensor_conv3 = batch_norm_layer(sensor_conv3, train, scope='sensor_BN3')
			sensor_conv3 = tf.nn.relu(sensor_conv3)
			sensor_conv3_shape = sensor_conv3.get_shape().as_list()
			sensor_conv_out = tf.reshape(sensor_conv3, [sensor_conv3_shape[0], sensor_conv3_shape[1], sensor_conv3_shape[2]*sensor_conv3_shape[3]])

		with tf.variable_scope("RNN"):
			gru_cell1 = tf.contrib.rnn.GRUCell(INTER_DIM)
			if train:
				gru_cell1 = tf.contrib.rnn.DropoutWrapper(gru_cell1, output_keep_prob=0.5)

			gru_cell2 = tf.contrib.rnn.GRUCell(INTER_DIM)
			if train:
				gru_cell2 = tf.contrib.rnn.DropoutWrapper(gru_cell2, output_keep_prob=0.5)

			cell = tf.contrib.rnn.MultiRNNCell([gru_cell1, gru_cell2])
			init_state = cell.zero_state(BATCH_SIZE, tf.float32)

			cell_output, final_stateTuple = tf.nn.dynamic_rnn(cell, sensor_conv_out, sequence_length=length, initial_state=init_state, time_major=False)

			sum_cell_out = tf.reduce_sum(cell_output*mask, axis=1, keep_dims=False)
			avg_cell_out = sum_cell_out/avgNum

		with tf.variable_scope("logits"):

			logits = layers.fully_connected(avg_cell_out, OUT_DIM, activation_fn=None, scope='output')

		return logits

def train():
	with tf.Graph().as_default():
		csvFileList = []
		csvDataFolder1 = os.path.join('data_process/sepHARData_'+select, "train")
		orgCsvFileList = os.listdir(csvDataFolder1)
		for csvFile in orgCsvFileList:
			if csvFile.endswith('.csv'):
				csvFileList.append(os.path.join(csvDataFolder1, csvFile))

		csvEvalFileList = []
		csvDataFolder2 = os.path.join('data_process/sepHARData_'+select, "eval")
		orgCsvFileList = os.listdir(csvDataFolder2)
		for csvFile in orgCsvFileList:
			if csvFile.endswith('.csv'):
				csvEvalFileList.append(os.path.join(csvDataFolder2, csvFile))

		global_step = tf.Variable(0, trainable=False)

		batch_feature, batch_label = input_pipeline(csvFileList, BATCH_SIZE)
		batch_eval_feature, batch_eval_label = input_pipeline(csvEvalFileList, BATCH_SIZE, shuffle_sample=False)

		# train_status = tf.placeholder(tf.bool)
		# trainX = tf.cond(train_status, lambda: tf.identity(batch_feature), lambda: tf.identity(batch_eval_feature))
		# trainY = tf.cond(train_status, lambda: tf.identity(batch_label), lambda: tf.identity(batch_eval_label))

		# logits = deepSense(trainX, train_status, name='deepSense')
		logits = deepSense(batch_feature, True, name='deepSense')

		# predict = tf.argmax(logits, axis=0)
			# batchLoss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=trainY)
			# batchLoss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=batch_label)
			# loss = tf.reduce_mean(batchLoss)
		with tf.name_scope('train_cross_entropy'):
			loss = loss_cross(logits, batch_label)
		tf.summary.scalar('train_cross entropy',loss)

		logits_eval = deepSense(batch_eval_feature, False, reuse=True, name='deepSense')
		# predict_eval = tf.argmax(logits_eval, axis=0)
		with tf.name_scope('eval_cross_entropy'):
			# loss_eval = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits_eval, labels=batch_eval_label))
			loss_eval = loss_cross(logits_eval, batch_eval_label)
		tf.summary.scalar('eval_cross entropy',loss_eval)

		with tf.name_scope('L2_loss'):
			t_vars = tf.trainable_variables()
			regularizers = 0.
			for var in t_vars:
				regularizers += tf.nn.l2_loss(var)
			l2_loss = loss + 5e-4 * regularizers
		tf.summary.scalar('L2_loss',loss)

		# optimizer = tf.train.RMSPropOptimizer(0.001)
		# gvs = optimizer.compute_gradients(loss, var_list=t_vars)
		# capped_gvs = [(tf.clip_by_norm(grad, 1.0), var) for grad, var in gvs]
		# discOptimizer = optimizer.apply_gradients(capped_gvs, global_step=global_step)
		# num_batches_per_epoch = TRAIN_SIZE / BATCH_SIZE
		# decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)
        #
		# # Decay the learning rate exponentially based on the number of steps.
		# lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE,
		# 								global_step,
		# 								decay_steps,
		# 								LEARNING_RATE_DECAY_FACTOR,
		# 								staircase=True)

		with tf.name_scope('train'):
			discOptimizer = tf.train.AdamOptimizer(
					learning_rate=INITIAL_LEARNING_RATE,
					beta1=0.5,
					beta2=0.9
				)
			optimizer = discOptimizer.minimize(l2_loss, var_list=t_vars)
			current_lr = (discOptimizer._lr_t * tf.sqrt(1 - discOptimizer._beta1_power) / (1 - discOptimizer._beta2_power))
			tf.summary.scalar('learning_rate', current_lr)

		with tf.name_scope('train_accuracy'):
			# p_label = np.argmax(batch_label, axis=0)
			# accuracy = np.mean(p_label == predict)
			correct = tf.nn.in_top_k(logits, batch_label, 1)
			correct = tf.cast(correct, tf.float64)
			accuracy = tf.reduce_mean(correct)
			tf.summary.scalar('accuracy', accuracy)

		with tf.name_scope('eval_accuracy'):
			# e_label = np.argmax(batch_eval_label, axis=0)
			# e_accuracy = np.mean(e_label == predict_eval)
			correct = tf.nn.in_top_k(logits_eval, batch_eval_label, 1)
			correct = tf.cast(correct, tf.float64)
			e_accuracy = tf.reduce_mean(correct)
			tf.summary.scalar('eval_accuracy', e_accuracy)

		saver = tf.train.Saver(tf.global_variables())

		summary_op = tf.summary.merge_all()

		with tf.Session() as sess:
			tf.global_variables_initializer().run()
			coord = tf.train.Coordinator()
			threads = tf.train.start_queue_runners(coord=coord)
			summary_writer = tf.summary.FileWriter(FLAGS.log_dir, graph=sess.graph)
			filename1 = 'train_log.txt'
			filename2 = 'eval_log.txt'
			with open(filename1,'w') as f1:
				f1.write("train_loss " + "train_accuracy " + "learning_rate\n")
			with open(filename2,'w') as f2:
				f2.write("eval_loss " + "eval_accuracy\n")

			for iteration in xrange(TOTAL_ITER_NUM):

				# _, lossV, _trainY, _predict = sess.run([discOptimizer, loss, trainY, predict], feed_dict = {
				# 	train_status: True
				# 	})
				_, lossV, train_accuracy, learning_rate = sess.run([optimizer, loss, accuracy, current_lr])
				#a0, bb1, bb2 = discOptimizer._lr, discOptimizer._beta1_power.eval(), discOptimizer._beta2_power.eval()
				#learning_rate = a0* (1-bb2)**0.5 /(1-bb1)
				with open(filename1,'a') as f1:
					f1.write(str(round(lossV, 5)) + " " + str(round(train_accuracy, 5)) + " " + str(round(learning_rate, 8)) + "\n")
				
				if iteration % 10 == 0:
					format_str = "step %d, train_loss = %.5f, train accuracy = %.2f%%"
					print (format_str % (iteration, lossV, train_accuracy * 100))
				# plot.plot('train cross entropy', lossV)
				# plot.plot('train accuracy', _accuracy)

				if iteration % 100 == 0:
					summary_str = sess.run(summary_op)
					summary_writer.add_summary(summary_str, iteration)


				if iteration % 500 == 0:
					dev_accuracy = []
					dev_cross_entropy = []
					for eval_idx in xrange(EVAL_ITER_NUM):
						# eval_loss_v, _trainY, _predict = sess.run([loss, trainY, predict], feed_dict ={train_status: False})
						eval_loss_v, eval_accuracy = sess.run([loss_eval, e_accuracy])
						dev_accuracy.append(eval_accuracy)
						dev_cross_entropy.append(eval_loss_v)

					format_str = "eval_step %d, eval_loss = %.5f, eval accuracy = %.2f%%"
					print (format_str % (iteration, np.mean(dev_cross_entropy), np.mean(dev_accuracy) * 100))

					with open(filename2,'a') as f2:
						f2.write(str(round(np.mean(dev_cross_entropy), 5)) + " " + str(round(np.mean(dev_accuracy), 5)) + "\n")

					# plot.plot('dev accuracy', np.mean(dev_accuracy))
					# plot.plot('dev cross entropy', np.mean(dev_cross_entropy))

				if iteration % 1000 == 0 or (iteration + 1) == TOTAL_ITER_NUM:
					checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
					saver.save(sess, checkpoint_path, global_step=iteration)

			summary_writer.close()


				# if (iteration < 5) or (iteration % 50 == 49):
				# 	plot.flush()
				#
				# plot.tick()

def main(argv=None):  # pylint: disable=unused-argument

	if gfile.Exists(FLAGS.train_dir):
	# gfile.DeleteRecursively(FLAGS.train_dir)
		pass
	else:
		gfile.MakeDirs(FLAGS.train_dir)
	if gfile.Exists(FLAGS.log_dir):
		# gfile.DeleteRecursively(FLAGS.train_dir)
		pass
	else:
		gfile.MakeDirs(FLAGS.log_dir)
	train()

if __name__ == '__main__':
	tf.app.run()
