
import numpy as np
import matplotlib.pyplot as plt
from LSTM import *

# Generate Data
X_t = np.arange(-70,10,0.1)

X_t = X_t.reshape(len(X_t),1)
Y_t = np.sin(X_t) + 0.1*np.random.randn(len(X_t),1) + np.exp((X_t + 20)*0.05)
# run model
dt   = 200
[lstm, dense1, dense2] = RunMyLSTM(Y_t, Y_t, n_neurons = 300,\
                                   n_epoch = 200, plot_each = 10, dt = dt,\
                                   momentum = 0.8, decay = 0.01,\
                                   learning_rate = 1e-5)

#output
Y_hat     = ApplyMyLSTM(Y_t,lstm, dense1, dense2)

#plot output   
X_plot     = np.arange(0,len(Y_t))
X_plot_hat = np.arange(0,len(Y_hat)) + dt

plt.plot(X_plot, Y_t)
plt.plot(X_plot_hat, Y_hat)
plt.legend(['y', '$\hat{y}$'])
plt.show()


