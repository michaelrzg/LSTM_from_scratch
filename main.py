import numpy as np
import matplotlib.pyplot as plot

X = np.arange(-70,10,0.1)
X = X.reshape(len(X),1)
Y = np.sin(X) + 0.1*np.random.randn(len(X),1) + np.exp((0.5*X + 20)*0.05)
plot.plot(X,Y)
plot.show()

from LSTM import *
lstm = LSTM(200,1)
lstm.forward_pass(X)

for h in lstm.H:
    plot.plot(np.arange(20),h[0:20],'k-',linewidth=1, alpha=0.05)
plot.show()

for f in lstm.F:
    plot.plot(np.arange(20),f[0:20],'k-',linewidth=2, alpha=.5)
plot.show()