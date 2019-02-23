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
		return self.w.T.dot(x)+ self.b

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
		x[x <= 0] = 0
		return x


# ADD other operations in ReLU if needed

# This is a class for a sigmoid layer followed by a cross entropy layer, the reason
# this is put into a single layer is because it has a simple gradient form
class SigmoidCrossEntropy(object):
	def __init__(self):
		#self.y = 0
		self.x = None
		self.L1_LT = None
		self.L2_LT = None
		self.ReLU = ReLU()

	def forward(self, x):
		# DEFINE forward function
		self.x = x
		self.net_1 = self.L1_LT.forward(x)
		self.a_1 = self.ReLU.forward(self.net_1)
		self.net_2 = self.L2_LT.forward(self.a_1.T)
		self.a_2 = self.sigmoid(self.net_2)
		if(self.a_2+0 != self.a_2):
			print("Error: \"NaN\" encountered")
			print("net 1: ",self.net_1)
			print("ReLU: ",self.a_1)
			print("net 2:",self.net_2)
			print("sigmoid:", self.a_2)
			exit(1)
		return 1 - self.a_2

	def sigmoid(self,x):
		return (1 / (1 + np.exp(-x)))

	def sigmoid_p(self,x):
		return (x * (1 - x))

	def cross(self,y,x):
		if x == 0 or x == 1:
			return (x-y)*(x-y)
		#return (x-y)*(x-y)
		return (y * np.log(x) + (1- y) * np.log(1-x))

	def cross_p(self,y,x):
		if x == 0 or x == 1:
			return (2 * (y-x))
		#return (2 * (y-x))
		return (x-y)/((x-1)*x)

	def backward(
		self,
		grad_output,
		learning_rate=0.0,
		momentum=0.0,
		l2_penalty=0.0
	):
		# DEFINE backward function
		partial_2 = self.L2_LT.backward(self.a_1).T.dot(
			self.cross_p(grad_output,self.a_2).dot(
				self.sigmoid_p(self.a_2)
			)
		)
		d_w2 = learning_rate * partial_2

		partial_1 = np.dot(self.x[np.newaxis].T,
			np.dot(
				self.cross_p(grad_output,self.a_2)*self.sigmoid_p(self.a_2),
				self.L2_LT.w.T
			)*(self.ReLU.backward(self.a_1))
		)
		d_w1 = learning_rate * partial_1
		#if (True in (d_w2 < 0)):
		#	print(d_w2)
		#	exit(1)
		#if (True in (d_w1 < 0)):
		#	print(d_w1)
		#	exit(1)
		self.L2_LT.w += d_w2
		self.L1_LT.w += d_w1


# ADD other operations and data entries in SigmoidCrossEntropy if needed
	#def set_y(self, y):
	#	self.y = y

	def set_parameters(self,hidden_units,dims):
		w1 = np.random.rand(dims,hidden_units)
		w2 = np.random.rand(hidden_units,1)
		b1 = np.random.rand(1,hidden_units)
		b2 = np.random.rand(1,1)
		self.L1_LT = LinearTransform(w1,b1)
		self.L2_LT = LinearTransform(w2,b2)


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
		error = 0
		for i in range(0,y_batch.shape[0]):
			#self.model.set_y(y_batch[i])
			y_hat = self.model.forward(x_batch[i])
			error = self.model.cross(y_batch[i],y_hat)
			self.model.backward(y_batch[i], learning_rate, momentum,l2_penalty,)
	def evaluate(self, x, y):
		# INSERT CODE for testing the network
		result = 0
		for i in range(0,y.shape[0]):
			output = self.model.forward(x[i])
			#print(output)
			if (y[i] == 1 and output > 0.5) or (y[i] == 0 and output <= 0.5):
				result += 1
			#print ([y[i], output, (y[i] > 0.5 and output > 0.5) or (y[i] < 0.5 and output < 0.5)], result)
		#print([result, y.shape[0],result/y.shape[0]])
		error = result/y.shape[0]
		return error, 1-error

# ADD other operations and data entries in MLP if needed
if __name__ == '__main__':
	if sys.version_info[0] < 3:
		data = pickle.load(open('cifar_2class_py2.p', 'rb'))
	else:
		data = pickle.load(open('cifar_2class_py2.p', 'rb'), encoding='bytes')
	print(data.get('train_data'))
	Lambda = 1.0
	if(len(sys.argv) > 1):
		print("Number of arguments detected: ", len(sys.argv))
		Lambda = float(sys.argv[1])
		print("Learning rate set to:", Lambda)
	norm = np.linalg.norm(data['train_data'])
	train_x = data['train_data']/norm
	train_y = data['train_labels']
	test_x = data['test_data']/norm
	test_y = data['test_labels']

	#Variables:
	train_loss = 0
	test_loss = 0
	train_accuracy = 0
	test_accuracy = 0

	#Hyper parameters:
	num_epochs = 20
	num_batches = 1000
	num_examples, input_dims = train_x.shape
	hidden_units = 5
	#Objects:
	mlp = MLP(input_dims, hidden_units)


	# INSERT YOUR CODE HERE
	# YOU CAN CHANGE num_epochs AND num_batches TO YOUR DESIRED VALUES

	for epoch in xrange(num_epochs):

		# INSERT YOUR CODE FOR EACH EPOCH HERE
		mlp.train(train_x, train_y, Lambda,0,0)
		train_loss,train_accuracy  = mlp.evaluate(train_x, train_y)
		test_loss,test_accuracy  = mlp.evaluate(test_x, test_y)
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
