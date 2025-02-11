import numpy as np
import matplotlib.pyplot as plt

def load_data(path):
    def one_hot(y):
        table = np.zeros((y.shape[0], 10))
        for i in range(y.shape[0]):
            table[i][int(y[i][0])] = 1 
        return table

    def normalize(x): 
        x = x / 255
        return x 

    data = np.loadtxt('{}'.format(path), delimiter = ',')
    return normalize(data[:,1:]),one_hot(data[:,:1])

def ReLU(x):
    return np.maximum(0,x)

def dReLU(x):
    return 1 * (x > 0) 

def softmax(z):
    z = z - np.max(z, axis = 1).reshape(z.shape[0],1)
    return np.exp(z) / np.sum(np.exp(z), axis = 1).reshape(z.shape[0],1)
    
def fully(x, W, b):
    assert x.shape[1] == W.shape[0]
    return x.dot(W) + b

def backwardFully(g, a):
    DW = np.dot(g.T, a).T
    db = np.sum(g, axis = 0)
    return (DW, db)

def update(W, b, lr, DW, db):
    assert DW.shape == W.shape
    W -= lr * DW
    assert db.shape == b.shape
    b -= lr * db

class NeuralNetwork:
    def __init__(self, X, y, batch = 64, lr = 1e-3,  epochs = 5):
        self.input = X 
        self.target = y
        self.batch = batch
        self.epochs = epochs
        self.lr = lr
        
        self.x = self.input[:self.batch] # batch input 
        self.y = self.target[:self.batch] # batch target value
        self.loss = []
        self.acc = []
        
        self.init_weights()
      
    def init_weights(self):
        self.W1 = np.random.randn(self.input.shape[1],256)
        self.W2 = np.random.randn(self.W1.shape[1],128)
        self.W3 = np.random.randn(self.W2.shape[1],self.y.shape[1])

        self.b1 = np.random.randn(self.W1.shape[1],)
        self.b2 = np.random.randn(self.W2.shape[1],)
        self.b3 = np.random.randn(self.W3.shape[1],)

    def shuffle(self):
        idx = [i for i in range(self.input.shape[0])]
        np.random.shuffle(idx)
        self.input = self.input[idx]
        self.target = self.target[idx]
        
    def feedforward(self):
        self.z1 = fully(self.x, self.W1, self.b1)
        self.a1 = ReLU(self.z1)

        self.z2 = fully(self.a1, self.W2, self.b2)
        self.a2 = ReLU(self.z2)

        self.z3 = fully(self.a2, self.W3, self.b3)
        self.a3 = softmax(self.z3)
        self.error = self.a3 - self.y

    def backprop(self):
        dcost = (1/self.batch)*self.error

        da3 = dcost # softmax 的反傳遞就等於 error 本身
        da2 = np.dot(da3, self.W3.T) * dReLU(self.z2)
        da1 = np.dot(da2, self.W2.T) * dReLU(self.z1)

        DW3, db3 = backwardFully(da3, self.a2)
        DW2, db2 = backwardFully(da2, self.a1)
        DW1, db1 = backwardFully(da1, self.x)

        update(self.W3, self.b3, self.lr, DW3, db3)
        update(self.W2, self.b2, self.lr, DW2, db2)
        update(self.W1, self.b1, self.lr, DW1, db1)

    def train(self):
        for epoch in range(self.epochs):
            l = 0
            acc = 0
            self.shuffle()
            print("epoch={}".format(epoch))
            
            for batch in range(self.input.shape[0]//self.batch-1):
                start = batch*self.batch
                end = (batch+1)*self.batch
                self.x = self.input[start:end]
                self.y = self.target[start:end]
                self.feedforward()
                self.backprop()
                l+=np.mean(self.error**2)
                acc+= np.count_nonzero(np.argmax(self.a3,axis=1) == np.argmax(self.y,axis=1)) / self.batch
                
            self.loss.append(l/(self.input.shape[0]//self.batch))
            self.acc.append(acc*100/(self.input.shape[0]//self.batch))
            
    def plot(self):
        plt.figure(dpi = 125)
        plt.plot(self.loss)
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.show()
    
    def acc_plot(self):
        plt.figure(dpi = 125)
        plt.plot(self.acc)
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        
    def test(self,xtest,ytest):
        self.x = xtest
        self.y = ytest
        self.feedforward()
        acc = np.count_nonzero(np.argmax(self.a3,axis=1) == np.argmax(self.y,axis=1)) / self.x.shape[0]
        print("Accuracy:", 100 * acc, "%")

print("start()")
print('load data....')
X_train, y_train = load_data('_data/train.csv')
X_test, y_test = load_data('_data/test.csv')
NN = NeuralNetwork(X_train, y_train) 
print("train()")
NN.train()
print("plot()")
NN.plot()
NN.test(X_test,y_test)
