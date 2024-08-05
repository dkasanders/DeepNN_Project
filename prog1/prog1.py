"""
Program 1 
By Dylan Kasanders
Data 471 WI24

This program implements a Neural Network class that can create an arbitrarily deep NN with regression and classification options. 
It includes three activation functions. (Sigmoid, TanH, and ReLU)

"""

import numpy as np;
import argparse as ap;
import time



class NeuralNetwork:
    def __init__(self, args):
        
        self.v = args.v # boolean value
        if args.train_feat != None:
            self.TRAIN_FEAT_FN = args.train_feat
        else:
            raise ValueError("No training feature file provided. See prog1.py --help for more info.")
        if args.train_target != None:
            self.TRAIN_TARGET_FN = args.train_target
        else:
            raise ValueError("No training target file provided. See prog1.py --help for more info.")
        if args.dev_feat != None:
            self.DEV_FEAT_FN = args.dev_feat
        else:
            raise ValueError("No dev feature file provided. See prog1.py --help for more info.")
        if args.dev_target != None:
            self.DEV_TARGET_FN = args.dev_target
        else:
            raise ValueError("No dev target file provided. See prog1.py --help for more info.")
        if args.epochs != None:
            if args.epochs > 0:
                self.EPOCHS = args.epochs
            else:
                raise ValueError("Number of epochs must be greater than zero.")
        else:
            raise ValueError("No number of epochs provided. See prog1.py --help for more info.")
        if args.learnrate != None:
            self.LEARNRATE = args.learnrate
        else:
            raise ValueError("No learnrate provided. See prog1.py --help for more info")
        if args.nunits != None:
            self.NUM_HIDDEN_UNITS = args.nunits
        else:
            self.NUM_HIDDEN_UNITS = 0
        if args.type != None:
            if args.type.upper() == 'C' or args.type.upper() == 'R':
                self.PROBLEM_MODE = args.type.upper()
            else:
                raise ValueError("Only classification ('C') and regression ('R') tasks supported.")
        else:
            raise ValueError("No problem type provided. See prog1.py --help for more info")

        if args.hidden_act != None:
            if args.hidden_act.upper() == 'SIG' or args.hidden_act.upper() == 'TANH' or args.hidden_act.upper() == 'RELU':
                self.HIDDEN_UNIT_ACTIVATION = args.hidden_act.upper()
            else:
                raise ValueError("Only sigmoid ('sig'), Tanh ('tanh'), and ReLU ('relu') activation functions supported.")
        else:
            self.HIDDEN_UNIT_ACTIVATION = ""
        if args.init_range != None:
            self.INIT_RANGE = args.init_range
        else:
            raise ValueError("No init range provided. See prog1.py --help for more info")
        if args.num_classes != None:
            self.C = args.num_classes
        else: 
            raise ValueError("No output layer dimension provided. See prog1.py --help for more info")
        if args.mb != None:
            self.MINIBATCH_SIZE = args.mb
        else:
            self.MINIBATCH_SIZE = 0 #Full batch training assumed
        if args.nlayers != None:
            self.NUM_HIDDEN_LAYERS = args.nlayers
        else:
            self.NUM_HIDDEN_LAYERS = 0


      

        

        self.x = np.loadtxt(self.TRAIN_FEAT_FN)
        self.x_target = np.loadtxt(self.TRAIN_TARGET_FN)
        self.dev_x = np.loadtxt(self.DEV_FEAT_FN)
        self.dev_x_target = np.loadtxt(self.DEV_TARGET_FN)

        #Initialize weight and bias vectors 
        self.w = []
        self.b = []
        if(self.NUM_HIDDEN_LAYERS == 0):
            self.w.append(np.random.uniform(-self.INIT_RANGE,self.INIT_RANGE, (self.x.shape[1], self.C)))
            self.b.append(np.random.uniform(-self.INIT_RANGE,self.INIT_RANGE, (1, self.C)))
        else:  
            for i in range((self.NUM_HIDDEN_LAYERS)): 
                if i == 0:
                    self.w.append(np.random.uniform(-self.INIT_RANGE,self.INIT_RANGE, (self.x.shape[1], self.NUM_HIDDEN_UNITS)))
                else:
                    self.w.append(np.random.uniform(-self.INIT_RANGE,self.INIT_RANGE, (self.NUM_HIDDEN_UNITS, self.NUM_HIDDEN_UNITS)))
                self.b.append(np.random.uniform(-self.INIT_RANGE,self.INIT_RANGE, (1, self.NUM_HIDDEN_UNITS)))
            
            self.w.append(np.random.uniform(-self.INIT_RANGE,self.INIT_RANGE, (self.NUM_HIDDEN_UNITS, self.C)))
            self.b.append(np.random.uniform(-self.INIT_RANGE,self.INIT_RANGE, (1, self.C)))



        

    #Main training function, uses minibatch size (if MB = 1 its SGD) to compute gradients.
    def Train(self):
        update = 0
        best_w = self.w
        best_b = self.b
        training_data = list(zip(self.x, self.x_target))
        if len(training_data) < self.MINIBATCH_SIZE:
            raise IndexError('MiniBatch size is greater than training data size.')
            exit
        for i in range(self.EPOCHS):
            np.random.shuffle(training_data)
            if (self.MINIBATCH_SIZE == 0):
                minibatches = [training_data]
            else:    
                minibatches = [training_data[i:i+self.MINIBATCH_SIZE] for i in range(0, len(training_data), self.MINIBATCH_SIZE) if len(training_data[i:i+self.MINIBATCH_SIZE]) == self.MINIBATCH_SIZE]
            for mb in minibatches:
                partials_w, partials_b = [], []
                for pair in mb:
                    partial_w, partial_b = self.BackProp(pair[0], pair[1])
                    partials_w.append(partial_w)
                    partials_b.append(partial_b)
                sum_w = partials_w[0].copy()
                for index, partial in enumerate(partials_w[1:]):
                    for column, colValue in enumerate(partial):
                        for row, rowValue in enumerate(colValue):
                            sum_w[column][row] = sum_w[column][row] + partials_w[index + 1][column][row]
                
                for column, colValue in enumerate(sum_w):
                    for row, rowValue in enumerate(colValue):
                        sum_w[column][row] = sum_w[column][row] / len(mb)

                sum_b = partials_b[0].copy()
                for index, partial in enumerate(partials_b[1:]):
                    for column, colValue in enumerate(partial):
                        for row, rowValue in enumerate(colValue):
                            sum_b[column][row] = sum_b[column][row] + partials_b[index + 1][column][row]

                for column, colValue in enumerate(sum_b):
                    for row, rowValue in enumerate(colValue):
                        sum_b[column][row] = sum_b[column][row] / len(mb)
                update = update + 1
                self.UpdateWeights(sum_w, sum_b)
                if self.v:
                    if self.PROBLEM_MODE == 'R':
                        train_loss, dev_loss = self.Evaluate()
                        if self.C == 1:
                            print("Update " + '{:06d}'.format(update) + ": train=" + '{:03f}'.format(train_loss[0][0]) + " dev="  + '{:03f}'.format(dev_loss[0][0]))
                        else:
                            print("Update " + '{:06d}'.format(update) + ": train=" + str(train_loss) + " dev=" + str(dev_loss))
                    else:
                        accuracy_train = self.accuracy(zip(self.x, self.x_target))
                        accuracy_dev = self.accuracy(zip(self.dev_x, self.dev_x_target))
                        print("Update " + '{:06d}'.format(update) + ": train=" + '{:03f}'.format(accuracy_train) + " dev=" + '{:03f}'.format(accuracy_dev))

            if self.PROBLEM_MODE == 'R':
                train_loss, dev_loss = self.Evaluate()
                if self.C == 1:
                    print("Epoch " + '{:03d}'.format(i + 1) + ": train=" + '{:03f}'.format(train_loss[0][0]) + " dev=" + '{:03f}'.format(dev_loss[0][0]))
                else: 
                    print("Epoch " + '{:03d}'.format(i + 1) + ": train=" + str(train_loss) + " dev=" + str(dev_loss))
            else:
                accuracy_train = self.accuracy(zip(self.x, self.x_target))
                accuracy_dev = self.accuracy(zip(self.dev_x, self.dev_x_target))
                print("Epoch " + '{:03d}'.format(i + 1) + ": train=" + '{:03f}'.format(accuracy_train) + " dev=" + '{:03f}'.format(accuracy_dev))

        
    #This evaluates the current model on the training set and the development set.             
    def Evaluate(self):
        train_loss = self.loss(zip(self.x, self.x_target))
        dev_loss = self.loss(zip(self.dev_x, self.dev_x_target))
        return train_loss, dev_loss
        

    #Updates weight and bias vectors given the partial derivatives
    def UpdateWeights(self, partial_w, partial_b):
        for index, weightColumn in enumerate(self.w):
            for columnIndex, weight in enumerate(weightColumn):
                self.w[index][columnIndex] = self.w[index][columnIndex] - self.LEARNRATE * partial_w[index][columnIndex]
        for index, biasColumn in enumerate(self.b):
            for biasIndex, bias in enumerate(biasColumn):
                self.b[index][biasIndex] = self.b[index][biasIndex] - self.LEARNRATE * partial_b[index][biasIndex]
            

    #Given a data point, it returns the gradients of the weight + bias vectors. 
    def BackProp(self, x, y):
        activationArr = [x]
        value = x
        zArray = [] #Array of all the values prior to the activation function 
        for w, b in zip(self.w, self.b):
            z = np.add(np.matmul(value, w), b)
            value = self.activationFunction(z)
            zArray.append(np.transpose(z))
            activationArr.append(np.transpose(value))

        if self.PROBLEM_MODE == 'R':
            cost = value - y
        elif self.PROBLEM_MODE == 'C':
            cost = np.array(value - self.OneHotVector(int(y)))
            
        else:
            raise ValueError("Invalid problem mode submitted")
            exit      
        partial_b = []
        partial_w = []
        if len(activationArr[-2].shape) == 1:
            activationArr[-2] = activationArr[-2].reshape(-1,1)
        partial_b.append(np.transpose(cost))
        partial_w.append(np.matmul(activationArr[-2], cost))
        layer = len(self.w) - 2
        while layer >= 0:
            cost = np.transpose(self.activationDerivative(zArray[layer]) * np.matmul(self.w[layer + 1], np.transpose(cost)))

            partial_b.append(cost)
            if len(activationArr[layer].shape) == 1:
                activationArr[layer] = activationArr[layer].reshape(-1, 1)  
            partial_w.append(np.matmul(activationArr[layer], cost))
            layer = layer - 1
        return partial_w[::-1], partial_b[::-1]
 

    #Converts a integer value x into one-hot-vector encoding based on the amount of classes. 
    def OneHotVector(self, x):
        if self.PROBLEM_MODE != 'C':
            raise ValueError("Functionality reserved for classification problems.")
            exit
        else:
            #For a dataset with C classes, the classes are encoded from 0 to C - 1.
            vector = np.zeros(self.C)
            vector[x] = 1
            return vector
    



    #Defines a loss for an amount of pairs.
    def loss(self, pairs):
        if(type(pairs) == zip):
            N = 0
            sum = 0 
            if(self.PROBLEM_MODE == 'R'): 
                for x, y in pairs:
                    N = N + 1
                    sum = sum + np.power((y - self.feedforward(x)), 2)
                sum = np.sqrt(sum)
                return sum/N
            elif(self.PROBLEM_MODE == 'C'):
                for x, y in pairs:
                    for k in range(self.C):
                        sum = sum + self.OneHotVector(y.astype(int)) * np.log(self.feedforward(x)[0][y.astype(int)]) 
                    N = N + 1
                return -sum/N
        else:
            #This means we have a tuple. This is necessary for SGD.
            x, y = pairs
            if(self.PROBLEM_MODE == 'R'):
                return np.sqrt(np.power((y - self.feedforward(x)), 2))
            elif(self.PROBLEM_MODE == 'C'):
                sum = 0 
                for k in range(self.C + 1):
                    sum = sum + self.OneHotVector(y.astype(int)) * np.log(self.feedforward(x)[0][y.astype(int)])
                return -sum 


    def activationFunction(self, n):
        if self.HIDDEN_UNIT_ACTIVATION == 'SIG': 
            return 1/(1 + np.exp(-n))
        elif self.HIDDEN_UNIT_ACTIVATION == 'TANH':
            return np.tanh(n)
        elif self.HIDDEN_UNIT_ACTIVATION == 'RELU':
            return np.maximum(0, n * 0.01) #overflow occurs when we don't add the mulitplier. 
        return n #Default case.
    
    def activationDerivative(self, n):
        if self.HIDDEN_UNIT_ACTIVATION == 'SIG':
            return self.activationFunction(n) * (1 - self.activationFunction(n))
        elif self.HIDDEN_UNIT_ACTIVATION == 'TANH':
            return 1 - (self.activationFunction(n) ** 2)
        elif self.HIDDEN_UNIT_ACTIVATION == 'RELU':
            if isinstance(n, np.ndarray):
                return np.where(n > 0, 1, 0)
            else:
                print(n)
                if(n > 0):
                    return 1
                return 0
        return 1 #Base case.

    
    def feedforward(self, x):
        value = x
        pairs = zip(self.w, self.b)
        for w, b in pairs:
            value = self.activationFunction(np.add(np.matmul(value, w), b))
        return value 

    """
    Helper method for measuring accuracy on the training set.
    Accuracy is defined as number of datapoints guessed correctly
    over the total amount of datapoints. A datapoint is guessed correctly
    if the model is most confident about the true correct answer.

    Due to the nature of this, this method is reserved for only
    classification models
    """
    def accuracy(self, set):
        if self.PROBLEM_MODE == 'C':
            
            correct = 0
            total = 0
            for x, y in set:
                total = total + 1
                model_output = self.feedforward(x)
                if(self.MaxIndex(model_output[0]) == int(y)):
                    correct = correct + 1
            
            return correct/total

    #Given an array, it returns the index the maximum is located at
    def MaxIndex(self, a):
        j = 0
        i = 1
        while i < len(a):
            if a[i] > a[j]:
                j = i
            i = i + 1
        return j 

        


parser = ap.ArgumentParser()
parser.add_argument("-v", help="Verbose mode", action="store_true")
parser.add_argument("-train_feat", help="Training feature filename", type=str)
parser.add_argument("-train_target", help="Training target filename", type=str)
parser.add_argument("-dev_feat", help="Development feature filename", type=str)
parser.add_argument("-dev_target", help="Development target filename", type=str)
parser.add_argument("-epochs", help="Epochs count for training", type=int)
parser.add_argument("-learnrate", help="Learning rate for training", type=float)
parser.add_argument("-nunits", help="Number of hidden units for hidden layers", type=int)
parser.add_argument("-type", help="Problem mode. 'C' or 'R'", type=str)
parser.add_argument("-hidden_act", help="Activation function for hidden layers. 'sig' or 'tanh' or 'relu'", type=str)
parser.add_argument('-init_range', help="Range for initializing weights and biases", type=float)
parser.add_argument('-num_classes', help="Size of the output layer", type=int)
parser.add_argument('-mb', help="Size of minibatch (0 if full-batch)", type=int)
parser.add_argument('-nlayers', help="Number of hidden layers", type=int)
args = parser.parse_args()



MyNN = NeuralNetwork(args)
MyNN.Train()
