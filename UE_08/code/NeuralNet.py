import numpy as np
import matplotlib.pyplot as plt

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
                #this is needed for initialization of later layers
                self.nodeNumber = size[0]
                #weight matrix
                self.W = np.random.normal(size=[size[0]+1, size[1]], scale=1E-4)
                #matrix of learn rates
                self.L = np.zeros((size[0]+1, size[1]))
                self.L.fill(0.1)
                #matrix for last gradient, important for calculating next learn rate
                self.lastGradient = np.ones((size[0]+1, size[1]))

        def forwardPropagate(self, X, Y):
                #this is important if only one example vector is supplied
                X = np.atleast_2d(X)
                #bias unit
                self.X = np.append(X, np.ones((X.shape[0], 1)), axis=1)
                #sets the true labels for error calculation afterwards
                self.errorLayer.setTrueLabels(np.atleast_2d(Y))
                #forward propagate activation values to next layer
                return self.nextLayer.forwardPropagate(self.X.dot(self.W))

        def forwardDream(self):
                self.errorLayer.setDream()
                return self.nextLayer.forwardPropagate(self.X.dot(self.W))

        def backwardPropagate(self, D):
                #calculates gradient and updates according to last learnrate
                gradient = (D.dot(self.X)).T
                self.W -= self.L * np.sign(gradient)
                
                #update learnrate
                self.L = np.array(map(learn_rate_line, self.lastGradient, gradient, self.L))                    
                self.lastGradient = gradient
                
                #update weights in next layer
                self.nextLayer.updateWeights() 

        def dream(self, D):
                gradient = self.W.dot(D).T
                print gradient
                raw_input()
                self.X -= 10 * gradient

        def predict(self, X, Y=None):
                #this is important if only one example vector is supplied
                X = np.atleast_2d(X)
                #bias unit
                self.X = np.append(X, np.ones((X.shape[0], 1)), axis=1)
                
                #set true labels in error layer only if supplied, for error calculation
                if (Y != None):
                        self.errorLayer.setTrueLabels(np.atleast_2d(Y))
                
                #propagate activation and information about error calculation to next layer
                return self.nextLayer.predict(self.X.dot(self.W), Y!=None)
        
        def setErrorLayer(self, errorLayer):
                #saves a reference to the error layer in order to set true labels
                self.errorLayer = errorLayer

class HiddenLayer:
        def __init__(self, size, activation=sigmoid):
                #this is needed for initialization of later layers
                self.nodeNumber = size[0]
                #weight matrix
                self.W = np.random.normal(size=[size[0]+1, size[1]], scale=1E-4)
                #matrix of learn rates
                self.L = np.zeros((size[0]+1, size[1]))
                self.L.fill(0.1)
                #matrix for last gradient, important for calculating next learn rate
                self.lastGradient = np.ones((size[0]+1, size[1]))
                self.activation = activation
        
        def forwardPropagate(self, S):
                #calculate activation and derivatives
                self.Z = self.activation(S)             
                self.Fp = self.activation(S, deriv=True).T
                #bias unit
                self.Z = np.append(self.Z, np.ones((self.Z.shape[0], 1)), axis=1)
                #propagate activation to next layer
                return self.nextLayer.forwardPropagate(self.Z.dot(self.W))
        
        def backwardPropagate(self, D):
                #caculate deltas for this layer according to dalte from last layer (not for bias!)
                self.D = self.W[0:-1, :].dot(D) * self.Fp
                #backpropagate D to last layer
                self.lastLayer.backwardPropagate(self.D)

        def dream(self, D):
                #weight_z = self.Z[:, 0:-1].T
                #weight_z = np.repeat(weight_z, self.W.shape[1], axis=1)
                #self.D = weight_z.dot(D) * self.Fp
                self.D = self.W[0:-1, :].dot(D) * self.Fp
                self.lastLayer.dream(self.D)
        
        def updateWeights(self):
                #calculates gradient and updates weight according to last learnrates
                gradient = (self.nextLayer.D.dot(self.Z)).T
                self.W -= self.L * np.sign(gradient)            
                #updates learnrates
                self.L = np.array(map(learn_rate_line, self.lastGradient, gradient, self.L))            
                #saves gradient for next step
                self.lastGradient = gradient            
                #weight update for next layer
                self.nextLayer.updateWeights() 
                
        def predict(self, X, calcError=False):
                #calculates activation and adds bias unit only, no derivation
                self.Z = self.activation(X)     
                self.Z = np.append(self.Z, np.ones((self.Z.shape[0], 1)), axis=1)
                #propagates activation and information about error calculation
                return self.nextLayer.predict(self.Z.dot(self.W), calcError)
                
class OutputLayer:
        def __init__(self, activation=sigmoid):
                self.activation = activation
        
        def forwardPropagate(self, S):          
                #this layer has no weights, only propagate activation
                return self.nextLayer.forwardPropagate(self.activation(S))
        
        def backwardPropagate(self, D):
                #need to transpose deltas coming from error layer
                self.D = D.T
                self.lastLayer.backwardPropagate(self.D)

        def dream(self, D): 
                self.D = D.T
                self.lastLayer.dream(self.D)
        
        def updateWeights(self):
                #the last hidden layer doesn't know it is the last, so it tries to call updateWeights on the output layer, therefore this stub is needed.
                return
        
        def predict(self, X, calcError=False):
                #calculates our prediction of the label or propagates it in order to get error sum.
                if not calcError:
                        return self.activation(X)
                else:
                        return self.nextLayer.forwardPropagate(self.activation(X), calcError)
        
        def setLastLayer(self, lastLayer):
                #method to set last layer for backward propagation
                self.lastLayer = lastLayer

class ErrorLayer:
        def __init__(self, error=quadratic_error):
                self.error = error
                self.dream = False
        
        def forwardPropagate(self, S, justError=False):
                #starts backward propagation by setting first D if justError is False, in both cases propagates error to next layer
                if self.dream:
                        newS = [0] * S.shape[1]
                        newS[np.argmax(S)] = 1
                        self.lastLayer.dream(S - newS)
                        return self.nextLayer.forwardPropagate(self.error(S, newS))
                elif not justError:
                        self.lastLayer.backwardPropagate(S - self.Y)
                return self.nextLayer.forwardPropagate(self.error(S, self.Y))
        
        def setNextLayer(self, nextLayer):
                #sets the next layer for forward propagation
                self.nextLayer = nextLayer
        
        def setTrueLabels(self, Y):
                #sets true labels for error calculation. called by input layer
                self.Y = Y

        def setDream(self):
                self.dream = True

class ErrorSum: 
        def forwardPropagate(self, S):  
                #returns the sum of squared errors, the end of forward propagation      
                return np.sum(S)

def connectLayers(first, second):
        #connects two layers so they can forward and backward propagate each other
        first.nextLayer = second
        second.lastLayer = first

class NeuralNet:
        def __init__(self, inputSize, outputSize, hiddenLayerConfig):           
                #adds the input layer
                self.inputLayer = InputLayer([inputSize, hiddenLayerConfig[0]])
                self.layers = [self.inputLayer]
                self.numberHidden = sum(hiddenLayerConfig)
                
                #adds the hidden layers up to the last
                for i,nodeNumber in enumerate(hiddenLayerConfig[:-2]):
                        self.layers.append(HiddenLayer([nodeNumber, hiddenLayerConfig[i+1]]))
                        connectLayers(self.layers[-2], self.layers[-1])
                
                #adds the last hidden layer
                self.layers.append(HiddenLayer([hiddenLayerConfig[-1], outputSize]))
                connectLayers(self.layers[-2], self.layers[-1])
                
                #adds the output layer
                self.layers.append(OutputLayer())
                connectLayers(self.layers[-2], self.layers[-1])
                
                #adds the error layer, connects the input layer with it, and adds the error sum layer to it
                self.layers.append(ErrorLayer())
                connectLayers(self.layers[-2], self.layers[-1])
                self.inputLayer.setErrorLayer(self.layers[-1])
                self.layers[-1].setNextLayer(ErrorSum())
        
        def train(self, train_data, train_labels, test_data, test_labels, difference=0.001):
                #trains our network until convergence (which is determined by small difference of error before and after training)
                last_error = float("Inf")
                error = 0
                
                while True:
                        error = self.inputLayer.forwardPropagate(train_data, train_labels)
                        if (error < last_error and last_error - error < difference):
                                break
                        last_error = error

                #predicts error for test set on trained network and gives us some benchmarks
                print("# hidden nodes: {0} // Training error: {1} // Test error: {2}".format(self.numberHidden, last_error, self.inputLayer.predict(test_data, test_labels)))

        def dream(self, dream_digit, difference=0.001):
                #dreams a digit until convergence (saving picture at intervals)
                last_error = float("Inf")
                error = 0
                counter = 0
                
                dream_digit = np.atleast_2d(dream_digit)
                self.inputLayer.X = np.append(dream_digit, np.ones((dream_digit.shape[0], 1)), axis=1)

                for i in range(1,100):
                        last_error = self.inputLayer.forwardDream()
                        picture = self.inputLayer.X

                
                print("Dreaming error: {0}\nDream: {1}".format(last_error, picture))
                plt.gray()
                plt.imshow(np.reshape(picture[:, 1:], (16,12)))
                plt.show()
