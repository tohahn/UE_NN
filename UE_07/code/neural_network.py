import numpy as np

def sigmoid(X, Y=None, deriv=False):
	if not deriv:
		return 1 / (1 + np.exp(-X))
	else:
		return sigmoid(X)*(1 - sigmoid(X))

def error(X, Y):
	return 0.5 * (X - Y)**2

class InputLayer:
	def __init__(self, size=None, activation=sigmoid):		
		self.activation = activation
		
		# Z is the matrix holding the activation values
		self.Z = None
		# W is the outgoing weight matrix for this layer
		self.W = None
		# S is the matrix that holds the inputs to this layer
		self.S = None

		if not is_error and not is_error_sum:
			self.size = size[0]-1
			self.W = np.random.normal(size=size, scale=1E-4)

	def forward_propagate(self):
		if self.is_input:
			return self.Z.dot(self.W)
		
		if self.is_error_sum:
			return np.sum(self.S)
		
		if self.is_error:
			self.S = sigmoid(self.S)
			self.D = (self.S - self.Y).T
		
		self.Z = self.activation(self.S, self.Y)
		
		if self.is_error:
			return self.Z
		
		self.Fp = self.activation(self.S, deriv=True).T
		# For hidden layers, we add the bias values here
		self.Z = np.append(self.Z, np.ones((self.Z.shape[0], 1)), axis=1)
		return self.Z.dot(self.W)
	
	def set_training_values(self, Y):
		self.Y = Y
	
	def set_training_data(self, X):
		self.Z = np.append(X, np.ones((X.shape[0], 1)), axis=1)


class NN:
	def __init__(self, layer_config):
		self.layers = []
		self.num_layers = len(layer_config)
		
		for i in range(self.num_layers-1):
			if i == 0:
				#addition of bias unit to input layer
				self.layers.append(Layer([layer_config[i]+1, layer_config[i+1]], is_input=True))
			else:
				#addition of bias unit to hidden layer(s)
				self.layers.append(Layer([layer_config[i]+1, layer_config[i+1]]))
		#error layer (error for every output neutron)
		self.layers.append(Layer(is_error=True, activation=error))
		#error sum layer (just sum of all errors)
		self.layers.append(Layer(is_error_sum=True))
		
		self.num_layers = len(self.layers)
		#our error layer, important for backpropagation
		self.error_layer = self.num_layers - 2
		
	def forward_propagate(self, X, Y):
		# initialize input and true data
		self.layers[0].set_training_data(X)
		self.layers[self.error_layer].set_training_values(Y)
		
		#forward propagation
		for i in range(self.num_layers-1):
			self.layers[i+1].S = self.layers[i].forward_propagate()
		return self.layers[-1].forward_propagate()

	def train(self, X, Y, eta):
		error = self.forward_propagate(X, Y)
		
		#backward propagation
		for i in range(self.error_layer-1, 0, -1):
			# We do not calculate deltas for the bias values
			W_nobias = self.layers[i].W[0:-1, :]
				
			self.layers[i].D = W_nobias.dot(self.layers[i+1].D) * self.layers[i].Fp
		
		#weight update
		for i in range(0, self.error_layer):
			W_grad = -eta*(self.layers[i+1].D.dot(self.layers[i].Z)).T
			self.layers[i].W += W_grad
		
		#return the error as a benchmark
		return error

	def evaluate(self, train_data, train_labels, test_data, test_labels, difference=0.001, eta=0.001):
		last_error = float("inf")
		while True:			
			error = self.train(train_data, train_labels, eta)
			if (error < last_error and last_error - error < difference):
				break
			last_error = error

		test_error = self.forward_propagate(test_data, test_labels)
		print("Anzahl verdeckter Knoten: {0} / Training error: {1:.5f} / Test error: {2:.5f}".format(self.layers[1].size, last_error, test_error))
