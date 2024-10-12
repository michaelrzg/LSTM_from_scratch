
import numpy as np
import matplotlib.pyplot as plt
from LSTM import *


# generate data
X = np.arange(-70,10,0.1)
X = X.reshape(len(X),1)
Y = np.sin(X) + 0.1*np.random.randn(len(X),1) + np.exp((0.5*X + 20)*0.05)


# of neurons or cells in our ltsm
n_neurons = 200
vector_length= max(X.shape)

#create ltsm and 2 dense layers to convert into predicition
lstm = LSTM(n_neurons)
dense1 = Dense(n_neurons,vector_length)
dense2 = Dense(vector_length,1)
n_epoch = 100

#create optimizers
optimizer_lstm = Optimizer_SGD_LSTM()
optimizer      = Optimizer_SGD()

#store loss for each iteration
loss_monitoring = np.zeros((100))

def run():
    for n in range(n_epoch):
        #forward pass
        lstm.forward_pass(X)
        H = np.array(lstm.short_term_memory)
        H = H.reshape((H.shape[0], H.shape[1]))
        
        dense1.forward(H[1:,:])
        dense2.forward(dense1.output)
        
        #output of forward pass
        Y_hat = dense2.output
        
        #simple loss
        dY = Y_hat - Y
        
        #Mean Squared Error Loss
        L = float(0.5* np.dot(dY.T,dY)/vector_length)
        
        #save loss value
        loss_monitoring[n] = L
        
        #backpropogation
        dense2.backward(dY)
        dense1.backward(dense2.dinputs)
        
        lstm.backward_pass(dense1.dinputs)
        
        # optimizer
        optimizer_lstm.pre_update_params()
        optimizer.pre_update_params()
        
        optimizer.update_params(dense1)
        optimizer.update_params(dense2)
        
        optimizer_lstm.update_params(lstm)
        
        optimizer_lstm.post_update_params()
        optimizer.post_update_params()
        
        #print current mean squared error
        print(f'current MSSE = {L: 0.3f}')

    #plot data
    plt.plot(range(n_epoch),loss_monitoring)
    plt.xlabel('epochs')
    plt.ylabel('MSSE')
    plt.yscale('log')
    plt.show()

run()
