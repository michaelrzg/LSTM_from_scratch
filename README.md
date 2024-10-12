# Summary

Long short-term memory (LSTM) is a type of recurrent neural network (RNN) that can process and retain information over multiple time steps. LSTMs are used in deep learning and artificial intelligence to learn, process, and classify sequential data, such as text, speech, and time series.
LSTMs are designed to prevent the neural network output from decaying or exploding as it cycles through feedback loops. This is called the vanishing gradient problem, which traditional RNNs face. LSTMs use gates to capture both short-term and long-term memory, and to regulate the flow of information into and out of the cell. The three gates are the input gate, the output gate, and the forget gate.
![image](https://github.com/user-attachments/assets/3b00d17d-4b4c-4be0-aebb-46a5709a2350)

# Walkthrough

Below is a high level walkthrough of how the code works.

## Phase 1: Forward pass

### Step 1: Initilization

First step in a LSTM model is initilizing our short term (h) and long term (C) weights, as well as our input and bias weights to np.zero. In our model, we initilize them both to np.zero values. For the first itteration, the values are randomized, but for each recurring pass, we use the previous output short term and long term weights. We also initilize our grad weights to random values.

```python
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

### Step 2: 'forget' Gate

Once we have our values initilzed, we can begin our first pass. We start by calculating the 'forget gate' which determines what ratio of our previous long term memory we are going to utilize for this pass.


To determine the forget gates outcome, we find the dot product of our input and our short term memory [t] times our input weights and our short term memory weight for at [t] and add or bias:

```python
# forget gate (determine how much of long term memory we will remember or 'forget')
# outout = input * weight + short term * short term weight + forget gate bias
outputf = np.dot(self.Uf, x) + np.dot(self.wf,ht) + self.bf
```

We then utilize sigmoid as an activation function to map the output of the forget gate to a value between 0 and 1, representing the ratio of our long term memory we are keeping. We save this ratio in our ft variable.

```python
# use our predefined array of sigmoid obje-cts
Sigmf[t].forward(outputf)
#grab output for later
ft = Sigmf[t].output
```

This value (ft) will be multiplied to our long term memory (C) in the next step.

### Step 3: Input Gate

The next step has 2 substeps: (1) calculating what percentage of our new value to add or 'remember' in our long term memory and (2) determining the actual value to add.


<p align = "center">
<img width="421" alt="Screenshot 2024-10-11 at 8 22 40 PM" src="https://github.com/user-attachments/assets/673de1db-7bed-4b7b-9490-df7419ea2777">
</p>

We start by calculating same liner combination as our forget gate but with out input gate.

```python
#repeat for input gate
outputi = np.dot(self.Ui, x) + np.dot(self.wi,ht) + self.bi
Sigmi[t].forward(outputi)
it = Sigmi[t].output
```

This output value from our sigmoid (it) determines the first substeps value (calculating what percentage of our new value to add or 'remember' in our long term memory). The next step requires a different activation function, the TanH, which returns a value between -1 and 1.

### Step 4: C_Tilde Gate and Updating Long Term Memory

We use the third gates, or the c_tilde gate value and multiply it with our input gates value to determine what value to add to our long term memory:

```python
# finally repeat for c-tilde
outc_tilde = np.dot(self.Ug,x) + np.dot(self.wg,ht) + self.bg
tan_1[t].forward(outc_tilde)
c_tilde = tan_1[t].output
```

Now we finally combine our values by multiplying our input gate value by our long term, then adding that to the product of our input gate and our ctilde gate:

```python
ct = np.multiply(ft,ct) + np.multiply(it,c_tilde)
```

Now out long term memort is updated.

### Step 5: Updating Shourt Term Memory

<p align="center">
<img width="438" alt="Screenshot 2024-10-11 at 8 31 50 PM" src="https://github.com/user-attachments/assets/cb9449dd-17a0-4af6-8e89-d2e6741f2705">
</p>

Finally, we update the short term memory to end this iteration and pass the values to the next iteration.
We update the Short Term Memory by multiplying the outcome of the output gate by the TanH of the long term memory.

```python
ht = np.multiply(tan_2[t].output,ot)
```

At this point, all thats left for our forward pass is to update our logs.

```python
# update our short (h) long (c) and ctilde logs
H[t+1] = ht
C[t+1] = ct
C_tilde[t] = c_tilde

#update logs of innter values
F[t] = ft
O[t] = ot
I[t] = it
```
