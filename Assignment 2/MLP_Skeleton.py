"""
Behnam Saeedi
Saeedib@oregonstate.edu
http://web.engr.oregonstate.edu/~saeedib/
"""

from __future__ import division
from __future__ import print_function
import numpy as np

import sys
try:
	import _pickle as pickle
except:
	import pickle

# This is a class for a LinearTransform layer which takes an input
# weight matrix W and computes W x as the forward step
class LinearTransform(object):
	def __init__(self, W, b):
		# DEFINE __init function
		self.w = W
		self.b = b

	def forward(self, x):
		# DEFINE forward function
		#print([x, self.w, self.b])
		#print(x)
		return np.dot(x,self.w) + self.b

	def backward(
		self,
		grad_output,
		learning_rate=0.0,
		momentum=0.0,
		l2_penalty=0.0,
	):
		# DEFINE backward function
		return grad_output

	def update_weights(self,w):
		self.w = w

# ADD other operations in LinearTransform if needed
# This is a class for a ReLU layer max(x,0)
class ReLU(object):
	def forward(self, x):
		# DEFINE forward function
		return np.maximum(x,0)

	def backward(
		self,
		grad_output,
		learning_rate=0.0,
		momentum=0.0,
		l2_penalty=0.0,
	):
		# DEFINE backward function
		x = np.copy(grad_output)
		x[x > 0] = 1
		x[x == 0] = 0
		x[x < 0] = 0
		return x


# ADD other operations in ReLU if needed

# This is a class for a sigmoid layer followed by a cross entropy layer, the reason
# this is put into a single layer is because it has a simple gradient form
class SigmoidCrossEntropy(object):
	def __init__(self):
		self.w1 = None
		self.w2 = None
		self.b1 = None
		self.b2 = None
		self.hidden = None
		self.output = None
		self.fwd = None
		self.x = None
		self.y_star = None
		self.L1_LT = None
		self.L2_LT = None
		self.ReLU = ReLU()


	def forward(self, x):
		# DEFINE forward function
		self.x = x
		self.hidden = self.ReLU.forward(self.L1_LT.forward(self.x))
		self.output = 1 / (1 + np.exp(-self.L2_LT.forward(self.hidden)))
		#print(-self.L2_LT.forward(self.hidden))
		#print(self.output)
		self.fwd = self.y_star * np.log(self.output)+(1-self.y_star)*np.log(1-self.output)
		return self.fwd

	def derivative(self,x):
		return ((self.y_star - 1)*np.exp(x) + self.y_star) /(np.exp(x)+1)

	def backward(
		self,
		grad_output,
		learning_rate=0.0,
		momentum=0.0,
		l2_penalty=0.0
	):
		# DEFINE backward function
		w2_error = self.y_star - self.fwd
#		print(self.output)
#		delta_w2 = w2_error * self.derivative(self.output)
#		print(self.L2_LT.forward(self.hidden))
#		print(self.w1[0])
		delta_w2 = w2_error * (self.output * (1 - self.output))
		w1_error = self.output * self.w2
		delta_w1 = w1_error * self.ReLU.backward(self.hidden)
		self.w2+= learning_rate *delta_w2 * self.hidden
		tmp_x = (np.copy(self.x))[np.newaxis]
		self.w1 += learning_rate * tmp_x.T.dot(delta_w1[np.newaxis])
		if self.w2[0]/1 != self.w2[0]:
			print("log:")
			print(self.output)
			print(w1_error)
			print(delta_w1)
			print(self.L1_LT.w)
			print(self.L1_LT.forward(self.x))
			print(self.hidden)
			print(-self.L2_LT.forward(self.hidden))
			print(self.output)
			exit(1)
		self.update_weights()


# ADD other operations and data entries in SigmoidCrossEntropy if needed
	def set_y(self, y_star):
		self.y_star = y_star

	def update_weights(self):
		self.L1_LT.update_weights(self.w1)
		self.L2_LT.update_weights(self.w2)

	def set_parameters(self,hidden_units,dims):
		self.w1 = np.random.rand(dims,hidden_units)
		self.w2 = np.random.rand(hidden_units)
		self.b1 = np.random.rand(hidden_units)
		self.b2 = np.random.rand(1)
		self.L1_LT = LinearTransform(self.w1,self.b1)
		self.L2_LT = LinearTransform(self.w2,self.b2)


# This is a class for the Multilayer perceptron
class MLP(object):
	def __init__(self, input_dims, hidden_units):
		# INSERT CODE for initializing the network
		self.dims = input_dims
		self.num_units = hidden_units
		self.model = SigmoidCrossEntropy()
		self.model.set_parameters(self.num_units, self.dims)


	def train(
		self,
		x_batch,
		y_batch,
		learning_rate,
		momentum,
		l2_penalty,
	):
		# INSERT CODE for training the network
		for i in range(0,y_batch.shape[0]):
			self.model.set_y(y_batch[i])
			self.model.forward(x_batch[i])
			self.model.backward(x_batch[i], learning_rate, momentum,l2_penalty,)
	def evaluate(self, x, y):
		# INSERT CODE for testing the network
		result = 0
		for i in range(0,y.shape[0]):
			output = self.model.forward(x[i])
			print(output)
			if (y[i] == 1 and output > 0.5) or (y[i] == 0 and output < 0.5):
				result += 1
			#print ([y[i], output, (y[i] > 0.5 and output > 0.5) or (y[i] < 0.5 and output < 0.5)], result)
		print([result, y.shape[0],result/y.shape[0]])
		return result/y.shape[0]

# ADD other operations and data entries in MLP if needed
if __name__ == '__main__':
	if sys.version_info[0] < 3:
		data = pickle.load(open('cifar_2class_py2.p', 'rb'))
	else:
		data = pickle.load(open('cifar_2class_py2.p', 'rb'), encoding='bytes')
	print(data.get('train_data'))
	norm = 1000000
	train_x = np.true_divide(data['train_data'],norm)
	train_y = data['train_labels']
	test_x = np.true_divide(data['test_data'],norm)
	test_y = data['test_labels']

	#Variables:
	train_loss = 0
	test_loss = 0
	train_accuracy = 0
	test_accuracy = 0

	#Hyper parameters:
	num_epochs = 10
	num_batches = 1000
	num_examples, input_dims = train_x.shape
	hidden_units = 5

	#Objects:
	mlp = MLP(input_dims, hidden_units)


	# INSERT YOUR CODE HERE
	# YOU CAN CHANGE num_epochs AND num_batches TO YOUR DESIRED VALUES

	for epoch in xrange(num_epochs):

		# INSERT YOUR CODE FOR EACH EPOCH HERE
		mlp.train(train_x, train_y, 0.0125,0,0)
		train_loss = mlp.evaluate(train_x, train_y)
		test_loss = mlp.evaluate(test_x, test_y)
		for b in xrange(num_batches):
			total_loss = 0.0
			# INSERT YOUR CODE FOR EACH MINI_BATCH HERE
			# MAKE SURE TO UPDATE total_loss
			print(
				'\r[Epoch {}, mb {}]	Avg.Loss = {:.3f}'.format(
					epoch + 1,
					b + 1,
					total_loss,
				),
				end='',
			)
			sys.stdout.flush()
		# INSERT YOUR CODE AFTER ALL MINI_BATCHES HERE
		# MAKE SURE TO COMPUTE train_loss, train_accuracy, test_loss, test_accuracy
		print()
		print('	Train Loss: {:.3f}	Train Acc.: {:.2f}%'.format(
				train_loss,
				100. * train_accuracy,
		))
		print('	Test Loss:  {:.3f}	Test Acc.:  {:.2f}%'.format(
				test_loss,
				100. * test_accuracy,
		))
