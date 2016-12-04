import numpy as np

def sigmoid(X, Y=None, deriv=False):
	if not deriv:
		return 1 / (1 + np.exp(-X))
	else:
		return sigmoid(X)*(1 - sigmoid(X))

def quadratic_error(X, Y):
	return 0.5 * (X - Y)**2

def learn_rate(last_grad, this_grad, last_l, up=1.2, down=0.5, l_min=1E-06, l_max=50.0):
	if (last_grad * this_grad > 0):
		return min(last_l*up, l_max)
	elif (last_grad * this_grad < 0):
		return max(last_l*down, l_min)
	return last_l	

def learn_rate_line(last_grad, this_grad, last_l):
	return map(learn_rate, last_grad, this_grad, last_l)

class InputLayer:
	def __init__(self, size):
		self.nodeNumber = size[0]
		self.W = np.random.normal(size=[size[0]+1, size[1]], scale=1E-4)
		self.L = np.zeros((size[0]+1, size[1]))
		self.L.fill(0.1)
		self.lastGradient = np.ones((size[0]+1, size[1]))

	def forwardPropagate(self, X, Y):
		X = np.atleast_2d(X)
		self.X = np.append(X, np.ones((X.shape[0], 1)), axis=1)
		self.errorLayer.setTrueLabels(np.atleast_2d(Y))
		return self.nextLayer.forwardPropagate(self.X.dot(self.W))
	
	def backwardPropagate(self, D):
		gradient = (D.dot(self.X)).T
		self.W -= self.L * np.sign(gradient)
		
		self.L = np.array(map(learn_rate_line, self.lastGradient, gradient, self.L))			
		self.lastGradient = gradient
		
		self.nextLayer.updateWeights() 
	
	def setErrorLayer(self, errorLayer):
		self.errorLayer = errorLayer

class HiddenLayer:
	def __init__(self, size, activation=sigmoid):
		self.nodeNumber = size[0]
		self.activation = activation		
		self.W = np.random.normal(size=[size[0]+1,size[1]], scale=1E-4)
		self.L = np.zeros((size[0]+1, size[1]))
		self.L.fill(0.1)
		self.lastGradient = np.ones((size[0]+1, size[1]))
	
	def forwardPropagate(self, S):		
		self.Z = self.activation(S)		
		self.Fp = self.activation(S, deriv=True).T
		self.Z = np.append(self.Z, np.ones((self.Z.shape[0], 1)), axis=1)
		return self.nextLayer.forwardPropagate(self.Z.dot(self.W))
	
	def backwardPropagate(self, D):
		self.D = self.W[0:-1, :].dot(D) * self.Fp
		self.lastLayer.backwardPropagate(self.D)
	
	def updateWeights(self):
		gradient = (self.nextLayer.D.dot(self.Z)).T
		self.W -= self.L * np.sign(gradient)
		
		self.L = np.array(map(learn_rate_line, self.lastGradient, gradient, self.L))
			
		self.lastGradient = gradient
		
		self.nextLayer.updateWeights() 

class OutputLayer:
	def __init__(self, activation=sigmoid):
		self.activation = activation
	
	def forwardPropagate(self, S):		
		return self.nextLayer.forwardPropagate(self.activation(S))
	
	def backwardPropagate(self, D):
		self.D = D.T
		self.lastLayer.backwardPropagate(self.D)
	
	def updateWeights(self):
		return
	
	def setLastLayer(self, lastLayer):
		self.lastLayer = lastLayer

class ErrorLayer:
	def __init__(self, error=quadratic_error):
		self.error = error
	
	def forwardPropagate(self, S):		
		self.lastLayer.backwardPropagate(S - self.Y)
		return self.nextLayer.forwardPropagate(self.error(S, self.Y))
	
	def setNextLayer(self, nextLayer):
		self.nextLayer = nextLayer
	
	def setTrueLabels(self, Y):
		self.Y = Y

class ErrorSum:	
	def forwardPropagate(self, S):		
		return np.sum(S)

def connectLayers(first, second):
	first.nextLayer = second
	second.lastLayer = first

class NeuralNet:
	def __init__(self, inputSize, outputSize, hiddenLayerConfig):		
		self.inputLayer = InputLayer([inputSize, hiddenLayerConfig[0]])
		self.layers = [self.inputLayer]
		
		for i,nodeNumber in enumerate(hiddenLayerConfig[:-2]):
			self.layers.append(HiddenLayer([nodeNumber, hiddenLayerConfig[i+1].nodeNumber]))
			connectLayers(self.layers[-2], self.layers[-1])
		
		self.layers.append(HiddenLayer([hiddenLayerConfig[-1], outputSize]))
		connectLayers(self.layers[-2], self.layers[-1])
		
		self.layers.append(OutputLayer())
		connectLayers(self.layers[-2], self.layers[-1])
		
		self.layers.append(ErrorLayer())
		connectLayers(self.layers[-2], self.layers[-1])
		self.inputLayer.setErrorLayer(self.layers[-1])
		self.layers[-1].setNextLayer(ErrorSum())
	
	def train(self, train_data, train_labels, test_data, test_labels, difference=0.001):
		last_error = float("Inf")
		error = 0
		
		while True:
			error = self.inputLayer.forwardPropagate(train_data, train_labels)
			if (error < last_error and last_error - error < difference):
				break
			last_error = error
		
		print("Error on test set: {0}".format(self.inputLayer.forwardPropagate(test_data, test_labels)))