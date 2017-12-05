
from PIL import Image
import argparse
import os.path
import sys
import time
import numpy as np
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES']= '0'

# Basic model parameters as external flags.
FLAGS = None

#image net
class Model():

	def __init__(self):
		""" Creates the model """
		self.def_directories()
		self.def_input()
		self.def_variables()
		self.def_params()
		#self.def_load_params()
		self.def_model()
		self.def_output()
		self.def_loss()
		self.def_metrics()
		self.def_optimizer()
		self.add_summaries()
		self.def_file_names()

	def _parse_function(self,proto):
		features = tf.parse_single_example(
		proto,
		features={
			'image1': tf.FixedLenFeature([], tf.string),
			'image2': tf.FixedLenFeature([], tf.string)
		})
		image1 = tf.decode_raw(features['image1'], tf.float64)
		image1 = tf.reshape(image1,(self.in_heigth_size,self.in_with_size,1))
		image1 = tf.cast(image1, tf.float32) 

		image2 = tf.decode_raw(features['image2'], tf.float64)
		image2 = tf.reshape(image2,(self.out_heigth_size,self.out_with_size,1))
		image2 = tf.cast(image2, tf.float32) 
		return image1, image2
	def def_directories(self):
		self.directory = sys.argv[0]+'_B:'+str(FLAGS.batch_size)+'_E:'+str(FLAGS.num_epochs)+'_L:'+str(FLAGS.learning_rate)+'_'+FLAGS.train_dir+'/'
		self.directory_npy = self.directory + 'NPY/'
		
		if not os.path.exists(self.directory):
			os.makedirs(self.directory)

		if not os.path.exists(self.directory_npy):
			os.makedirs(self.directory_npy)

	def def_file_names(self):
		self.TRAIN_FILE = 'train.tfrecords'
		self.VALIDATION_FILE = 'validation.tfrecords'
		self.TEST_FILE = 'test.tfrecords'

	def def_variables(self):
		self.size_after_conv = int(np.ceil(float(FLAGS.height)/float(2**3))) * int(np.ceil(float(FLAGS.x_width)/float(2**3)))

	def def_input(self):
		self.in_heigth_size = FLAGS.height
		self.in_with_size = FLAGS.x_width
		self.out_heigth_size = FLAGS.height
		self.out_with_size = FLAGS.y_width

		self.Xa = tf.placeholder(tf.float32, [None, self.in_heigth_size,self.in_with_size,1])
		self.Xb = tf.placeholder(tf.float32, [None, self.in_heigth_size,self.in_with_size,1])
		self.Y_true = tf.placeholder(tf.float32, [None, self.out_heigth_size, self.out_with_size,1])

		self.labels = tf.placeholder(tf.int32, [None, None])

	def def_params(self):
		with tf.name_scope('params'):
			self.weight = {}
			self.bias = {}
			
			with tf.name_scope('params'):
				with tf.name_scope('conv1'):
					self.weight["W_cn1a"] = self.weight_variable([5, 5, 1, 48])
					self.weight["W_cn1b"] = self.weight_variable([5, 5, 1, 48])
					self.bias["b_cn1a"] = self.bias_variable([48])
					self.bias["b_cn1b"] = self.bias_variable([48])

				with tf.name_scope('conv2'):
					self.weight["W_cn2a"] = self.weight_variable([3, 3, 48, 128])
					self.weight["W_cn2b"] = self.weight_variable([3, 3, 48, 128])
					self.bias["b_cn2a"] = self.bias_variable([128])
					self.bias["b_cn2b"] = self.bias_variable([128])

				with tf.name_scope('conv3'):
					self.weight["W_cn3a"] = self.weight_variable([3, 3, 128, 192])
					self.weight["W_cn3b"] = self.weight_variable([3, 3, 128, 192])
					self.bias["b_cn3a"] = self.bias_variable([192])
					self.bias["b_cn3b"] = self.bias_variable([192])

				with tf.name_scope('conv4'):
					self.weight["W_cn4a"] = self.weight_variable([3, 3, 192, 192])
					self.weight["W_cn4b"] = self.weight_variable([3, 3, 192, 192])
					self.bias["b_cn4a"] = self.bias_variable([192])
					self.bias["b_cn4b"] = self.bias_variable([192])

				with tf.name_scope('conv5'):
					self.weight["W_cn5a"] = self.weight_variable([3, 3, 192, 128])
					self.weight["W_cn5b"] = self.weight_variable([3, 3, 192, 128])
					self.bias["b_cn5a"] = self.bias_variable([128])
					self.bias["b_cn5b"] = self.bias_variable([128])

				with tf.name_scope('fc1'):
					self.weight["W_fc1"] = self.weight_variable([self.size_after_conv * 256, self.out_with_size * self.out_heigth_size])
					self.bias["b_fc1"] = self.bias_variable([self.out_with_size * self.out_heigth_size])

	def conv2d(self,x,W):
		return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

	def max_pool_2x2(self,x):
		return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME')

	def weight_variable(self,shape):
		initializer = tf.contrib.layers.xavier_initializer()(shape)
		return tf.Variable(initializer)
	def bias_variable(self,shape):
		initializer = tf.contrib.layers.xavier_initializer()(shape)
		return tf.Variable(initializer)

	def def_model(self):
		with tf.name_scope('model'):
			with tf.name_scope('conv1'):
				h_cn1a = tf.nn.relu(self.conv2d(self.Xa, self.weight["W_cn1a"]) + self.bias["b_cn1a"])
				h_cn1b = tf.nn.relu(self.conv2d(self.Xb, self.weight["W_cn1b"]) + self.bias["b_cn1b"])

			with tf.name_scope('pool1'):
				h_pool1a = self.max_pool_2x2(h_cn1a)
				h_pool1b = self.max_pool_2x2(h_cn1b)

			with tf.name_scope('conv2'):
				h_cn2a = tf.nn.relu(self.conv2d(h_pool1a, self.weight["W_cn2a"]) + self.bias["b_cn2a"])
				h_cn2b = tf.nn.relu(self.conv2d(h_pool1b, self.weight["W_cn2b"]) + self.bias["b_cn2b"])

			with tf.name_scope('pool2'):
				h_pool2a = self.max_pool_2x2(h_cn2a)
				h_pool2b = self.max_pool_2x2(h_cn2b)

			with tf.name_scope('conv3'):
				h_cn3a = tf.nn.relu(self.conv2d(h_pool2a, self.weight["W_cn3a"]) + self.bias["b_cn3a"])
				h_cn3b = tf.nn.relu(self.conv2d(h_pool2b, self.weight["W_cn3b"]) + self.bias["b_cn3b"])

			with tf.name_scope('conv4'):
				h_cn4a = tf.nn.relu(self.conv2d(h_cn3a, self.weight["W_cn4a"]) + self.bias["b_cn4a"])
				h_cn4b = tf.nn.relu(self.conv2d(h_cn3b, self.weight["W_cn4b"]) + self.bias["b_cn4b"])

			with tf.name_scope('conv5'):
				h_cn5a = tf.nn.relu(self.conv2d(h_cn4a, self.weight["W_cn5a"]) + self.bias["b_cn5a"])
				h_cn5b = tf.nn.relu(self.conv2d(h_cn4b, self.weight["W_cn5b"]) + self.bias["b_cn5b"])

			with tf.name_scope('pool5'):
				h_pool5a = self.max_pool_2x2(h_cn5a)
				h_pool5b = self.max_pool_2x2(h_cn5b)
				h_pool5 = tf.concat([h_pool5a, h_pool5b], axis=3)

			with tf.name_scope('fc1'):
				h_pool5_flat = tf.reshape(h_pool5, [-1, self.size_after_conv * 256])
				Y_pred = tf.matmul(h_pool5_flat, self.weight["W_fc1"]) + self.bias["b_fc1"]
				print(Y_pred)
			
		self.Y_pred = tf.reshape(Y_pred,(FLAGS.batch_size,self.out_heigth_size, self.out_with_size,1))

	def def_output(self):
		with tf.name_scope('output'):
			self.label_pred = self.Y_pred
			self.label_true = self.Y_true

	def def_loss(self):
		with tf.name_scope('loss'):
			self.loss = tf.reduce_mean(tf.squared_difference(self.Y_true, self.Y_pred)) * 10

	def def_optimizer(self):
		with tf.name_scope('optimizer'):
			grad = tf.train.GradientDescentOptimizer(learning_rate=FLAGS.learning_rate)
  			self.optimizer = grad.minimize(self.loss)

	def def_metrics(self):
		""" Adds metrics """
		with tf.name_scope('metrics'):
			cmp_Y_true = tf.equal(tf.round(self.label_true * 10), tf.round(self.label_pred * 10))
			self.accuracy = tf.reduce_mean(tf.cast(cmp_Y_true, tf.float32), name='accuracy')

	def add_summaries(self):
		with tf.name_scope('summaries'):
			tf.summary.scalar('loss', self.loss)
			tf.summary.scalar('accuracy', self.accuracy)
		self.summary = tf.summary.merge_all()

	def train(self):

		# Training Dataset
		train_dataset = tf.contrib.data.TFRecordDataset([FLAGS.train_dir+'/'+self.TRAIN_FILE])
		# Parse the record into tensors.
		train_dataset = train_dataset.map(self._parse_function)
		train_dataset = train_dataset.shuffle(buffer_size=10000)
		train_dataset = train_dataset.batch(FLAGS.batch_size)
		# Training Dataset
		validation_dataset = tf.contrib.data.TFRecordDataset([FLAGS.train_dir+'/'+self.TEST_FILE])
		# Parse the record into tensors.
		validation_dataset = validation_dataset.map(self._parse_function)
		validation_dataset = validation_dataset.shuffle(buffer_size=10000)
		validation_dataset = validation_dataset.batch(FLAGS.batch_size)

		handle = tf.placeholder(tf.string, shape=[])
		iterator = tf.contrib.data.Iterator.from_string_handle(handle,train_dataset.output_types, train_dataset.output_shapes)
		iterator2 = tf.contrib.data.Iterator.from_string_handle(handle,validation_dataset.output_types, validation_dataset.output_shapes)
		next_element = iterator.get_next()
		next_element2 = iterator2.get_next()
		training_iterator = train_dataset.make_initializable_iterator()
		validation_iterator = validation_dataset.make_initializable_iterator()


		with tf.Session() as sess:

			train_writer = tf.summary.FileWriter('graphs/'+sys.argv[0]+'_B:'+str(FLAGS.batch_size)+'_E:'+str(FLAGS.num_epochs)+'_L:'+str(FLAGS.learning_rate)+"_"+FLAGS.train_dir+"_train")
			test_writer = tf.summary.FileWriter('graphs/'+sys.argv[0]+'_B:'+str(FLAGS.batch_size)+'_E:'+str(FLAGS.num_epochs)+'_L:'+str(FLAGS.learning_rate)+"_"+FLAGS.train_dir+"_test")
			train_writer.add_graph(sess.graph)

			init_op = tf.group(tf.global_variables_initializer(),tf.local_variables_initializer())
			sess.run(init_op)
			training_handle = sess.run(training_iterator.string_handle())
			validation_handle = sess.run(validation_iterator.string_handle())
			# Compute for 10 epochs.
			step = FLAGS.init_step
			for i in range(FLAGS.num_epochs):

				sess.run(training_iterator.initializer)
				while True:
					try:
						x_, y_ = sess.run(next_element,feed_dict={handle: training_handle})

						for key, value in self.bias.iteritems():
							bias_list.append(self.bias[key])
						fetches = [self.optimizer,self.accuracy,self.loss,self.summary]
						results = sess.run(fetches, feed_dict={self.Xa: x_,self.Xb: x_, self.Y_true: y_})
						optimizer_ = results[0]
						accuracy_ = results[1]
						loss_ = results[2]
						summary_ = results[3]
						msg = "Train: {:3d} loss: ({:6.2f}), acc: ({:6.2f})"
						msg = msg.format(step, loss_, accuracy_)
						train_writer.add_summary(summary_, step)
						step += 1
					except tf.errors.OutOfRangeError:
						break
				sess.run(validation_iterator.initializer)
				#TEST
				x_, y_ = sess.run(next_element2,feed_dict={handle: validation_handle})
				fetches = [self.optimizer,self.accuracy,self.loss,self.summary]
				_, train_acc_,train_loss_,summary_ = sess.run(fetches,  feed_dict={self.Xa: x_,self.Xb: x_, self.Y_true: y_})
				msg = "Test: {:3d} loss: ({:6.2f}), acc: ({:6.2f})"
				msg = msg.format(step, train_loss_, train_acc_)
				print(msg)
				test_writer.add_summary(summary_, step)
				

def run():

	# defines our model
	model = Model()
	# trains our model
	model.train()

def main(args):
	run()
	return 0


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument(
		'--learning_rate',
		type=float,
		default=0.01,
		help='Initial learning rate.'
	)
	parser.add_argument(
		'--num_epochs',
		type=int,
		default=2,
		help='Number of epochs to run trainer.'
	)

	parser.add_argument(
		'--batch_size',
		type=int,
		default=100,
		help='Batch size.'
	)
	parser.add_argument(
		'--train_dir',
		type=str,
		default='/tmp/data',
		help='Directory with the training data.'
	)
	
	parser.add_argument(
		'--height',
		type=int,
		default=256,
		help='height of the image'
	)
	parser.add_argument(
		'--x_width',
		type=int,
		default=20,
		help='width of the in window'
	)
	parser.add_argument(
		'--y_width',
		type=int,
		default=5,
		help='width of the out window'
	)


	FLAGS, unparsed = parser.parse_known_args()
	tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)