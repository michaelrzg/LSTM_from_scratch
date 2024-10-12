# Summary
Long short-term memory (LSTM) is a type of recurrent neural network (RNN) that can process and retain information over multiple time steps. LSTMs are used in deep learning and artificial intelligence to learn, process, and classify sequential data, such as text, speech, and time series. 
LSTMs are designed to prevent the neural network output from decaying or exploding as it cycles through feedback loops. This is called the vanishing gradient problem, which traditional RNNs face. LSTMs use gates to capture both short-term and long-term memory, and to regulate the flow of information into and out of the cell. The three gates are the input gate, the output gate, and the forget gate. 
![image](https://github.com/user-attachments/assets/3b00d17d-4b4c-4be0-aebb-46a5709a2350)
# Walkthrough
Below is a high level walkthrough of how the code works.
## Phase 1: Forward pass
### Step 1
First step in a LSTM model is initilizing our short term (h) and long term (C) weights, as well as our input and bias weights to np.zero. In our model, we initilize them both to np.zero values. For the first itteration, the values are randomized, but for each recurring pass, we use the previous output short term and long term weights. We also initilize our grad weights to random values.

``` python
 # set up lists to store for short term, long term, and c_tilde (last gate) states at timestamps 0 to t
self.short_term_memory = [np.zeros((self.n_neurons,1)) for x in range(self.max_vector+1)]
self.long_term_memory  = [np.zeros((self.n_neurons,1)) for x in range(self.max_vector+1)]
self.update_gate    = [np.zeros((self.n_neurons,1)) for x in range(self.max_vector)]

# forget gate values
self.dUf = 0.1*np.random.rand(self.n_neurons,1)
self.dbf = 0.1*np.random.rand(self.n_neurons,1)
self.dwf = 0.1*np.random.rand(self.n_neurons,self.n_neurons)

# input gate values
self.dUi = 0.1*np.random.rand(self.n_neurons,1)
self.dbi = 0.1*np.random.rand(self.n_neurons,1)
self.dwi = 0.1*np.random.rand(self.n_neurons,self.n_neurons)

#output gate values
self.dUo = 0.1*np.random.rand(self.n_neurons,1)
self.dbo = 0.1*np.random.rand(self.n_neurons,1)
self.dwo = 0.1*np.random.rand(self.n_neurons,self.n_neurons)

# update gate
self.dUg = 0.1*np.random.rand(self.n_neurons,1)
self.dbg = 0.1*np.random.rand(self.n_neurons,1)
self.dwg = 0.1*np.random.rand(self.n_neurons,self.n_neurons)
```

Once we have 


