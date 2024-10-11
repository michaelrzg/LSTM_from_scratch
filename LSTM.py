# Long Short Term Memory Model from Scratc
# Michael Rizig

import numpy as np

# sigmoid activation function
class Sigmoid_Activation():
    
    # for forward pass usage
    def forward(self, x):
        #calculate sigmoid of x
        sigx = 1/(1+np.exp(-x))
        #store
        self.output = self.inputs = sigx
    
    # for back prop usage
    def backwards(self,outer_derivatives):    
        # we need to multuply outer derivatives by the inner derivative (chain rule stuff)
        # find inner derivatives
        dx = np.multiply(self.inputs,(1-self.inputs))
        # store x * dx
        self.dinputs = np.multiply(dx,outer_derivatives)


# sigmoid activation function
class Tanh_Activation():
    # for forward pass usage
    def forward(self, x):
        #calculate tanh (with numpy)
        self.output = np.tanh(x)
        # store
        self.inputs = x
    # for back prop usage
    def backwards(self,outer_derivatives):
        # we need to multuply outer derivatives by the inner derivative (chain rule stuff)
        # calc inner derivative
        dx = 1- (self.output**2)
        # store x * dx
        self.dinputs = np.multiply(dx,outer_derivatives)

class Dense():
    
    def __init__(self, n_inputs, nodes):
        self.weights = 0.1*np.random.randn(n_inputs,nodes)
        self.bias = np.zeros((1,nodes))
    #forward propogation
    def forward(self,inputs):
        self.output = np.dot(inputs,self.weights) + self.bias
        self.inputs = inputs
    #back propogation
    def backward(self,dvalues):
        self.dweights = np.dot(self.inputs.T,dvalues)
        self.dbias = np.sum(dvalues,axis=0,keepdims=True)
        self.dinputs = np.dot(dvalues,self.weights.T)

class LSTM():     
    def __init__(self, nodes) -> None:
        # num of n_neurons
        self.n_neurons = nodes
        # num of features
        self.n_features = 1
        # learnables
        # forget gate values
        self.Uf = 0.1*np.random.rand(nodes,1)
        self.bf = 0.1*np.random.rand(nodes,1)
        self.wf = 0.1*np.random.rand(nodes,nodes)

        # input gate values
        self.Ui = 0.1*np.random.rand(nodes,1)
        self.bi = 0.1*np.random.rand(nodes,1)
        self.wi = 0.1*np.random.rand(nodes,nodes)

        #output gate values
        self.Uo = 0.1*np.random.rand(nodes,1)
        self.bo = 0.1*np.random.rand(nodes,1)
        self.wo = 0.1*np.random.rand(nodes,nodes)

        #output gate values
        self.Uo = 0.1*np.random.rand(nodes,1)
        self.bo = 0.1*np.random.rand(nodes,1)
        self.wo = 0.1*np.random.rand(nodes,nodes)

        # c tilde
        self.Ug = 0.1*np.random.rand(nodes,1)
        self.bg = 0.1*np.random.rand(nodes,1)
        self.wg = 0.1*np.random.rand(nodes,nodes)

    def forward_pass(self,data):
        # find vector length
        self.max_vector = max(data.shape)
        # save data in class
        self.data = data
        # set up lists to store for short term, long term, and c_tilde (last gate) states at timestamps 0 to t
        self.short_term_memory = [np.zeros((self.n_neurons,1)) for x in range(self.max_vector+1)]
        self.long_term_memory  = [np.zeros((self.n_neurons,1)) for x in range(self.max_vector+1)]
        self.update_gate    = [np.zeros((self.n_neurons,1)) for x in range(self.max_vector)]

        # set up lists to store gate inner values at points 0-(t-1) (for debugging later)
        self.forget_gate = [np.zeros((self.n_neurons,1)) for x in range(self.max_vector)]
        self.output_gate  = [np.zeros((self.n_neurons,1)) for x in range(self.max_vector)]
        self.input_gate = [np.zeros((self.n_neurons,1)) for x in range(self.max_vector)]    

        #store derivative (gradients) of gate values (delta of learnables)

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

        # c tilde
        self.dUg = 0.1*np.random.rand(self.n_neurons,1)
        self.dbg = 0.1*np.random.rand(self.n_neurons,1)
        self.dwg = 0.1*np.random.rand(self.n_neurons,self.n_neurons)

        # set up activation functions for each gate
        # since each activation function has its own values to store, we need objects
        forget_gate_sigmoid = [Sigmoid_Activation() for x in range(self.max_vector)]
        input_gate_sigmoid = [Sigmoid_Activation() for x in range(self.max_vector)]
        output_gate_sigmoid = [Sigmoid_Activation() for x in range(self.max_vector)]

        #we have 2 tan gates in input and output

        tan_1 = [Tanh_Activation() for x in range(self.max_vector)]
        tan_2 = [Tanh_Activation() for x in range(self.max_vector)]

        # set our initial outer and inner memories at t=0
        ht = self.short_term_memory[0]
        ct = self.long_term_memory[0]
        # run cell to progress t and  generate values
        [H, C,self.Sigmf,self.Sigmi, self.Sigmo, self.tan_1, self.tan_2,F,O,I,C_tilde] =self.lstm_cell(
            data,ht,ct,forget_gate_sigmoid,input_gate_sigmoid,output_gate_sigmoid,tan_1,tan_2,self.short_term_memory,self.long_term_memory,self.forget_gate,self.output_gate,
            self.input_gate,self.update_gate
        )

        # save outputs of cell
        self.forget_gate = F
        self.output_gate = O
        self.input_gate = I
        self.update_gate = C_tilde
        #save short and long term memory
        self.short_term_memory=H
        self.long_term_memory = C
        # save activaton function states
        self.Sigf = forget_gate_sigmoid
        self.Sigi = input_gate_sigmoid
        self.Sigo = output_gate_sigmoid
        self.Tan1 = tan_1
        self.Tan2 = tan_2

    def lstm_cell(self,data,ht,ct,Sigmf,Sigmi,Sigmo,tan_1,tan_2,H,C,F,O,I,C_tilde ):
        # for each datapoint time stamp
        for t, x in enumerate(data):
            # reshape so we can use dot
            x = x.reshape(1,1)

            # forget gate (determine how much of long term memory we will remember or 'forget')
            # outout = input * weight + short term * short term weight + forget gate bias
            outputf = np.dot(self.Uf, x) + np.dot(self.wf,ht) + self.bf
            # take sigmoid to return value between 0 and 1
            # use our predefined array of sigmoid obje-cts 
            Sigmf[t].forward(outputf)
            #grab output for later
            ft = Sigmf[t].output

            #repeat for input gate
            outputi = np.dot(self.Ui, x) + np.dot(self.wi,ht) + self.bi
            Sigmi[t].forward(outputi)
            it = Sigmi[t].output

            #repeat for output gate
            outputo = np.dot(self.Uo, x) + np.dot(self.wo,ht) + self.bo
            Sigmo[t].forward(outputo)
            ot = Sigmo[t].output

            # finally repeat for c-tilde
            outc_tilde = np.dot(self.Ug,x) + np.dot(self.wg,ht) + self.bg
            tan_1[t].forward(outc_tilde)
            c_tilde = tan_1[t].output

            # combine input gate and forget gate to add to long term memory
            # multiply forget with long term to determine remember ration
            # multiply input with c_tilde (last gate) to determine how much we use new data
            # add new data to long term
            ct = np.multiply(ft,ct) + np.multiply(it,c_tilde)
            # pass this new long term to tanh to determine short term by mutliplying it with output of sig act
            # sig activ (0 to 1) * tan activ(-1 to 1)
            tan_2[t].forward(ct)
            ht = np.multiply(tan_2[t].output,ot)

            # update our short (h) long (c) and ctilde logs 
            H[t+1] = ht
            C[t+1] = ct
            C_tilde[t] = c_tilde
            
            #update logs of innter values
            F[t] = ft
            O[t] = ot
            I[t] = it
            # return values
        return (H,C,Sigmf,Sigmi,Sigmo,tan_1,tan_2,F,O,I,C_tilde)

    def backward_pass(self,dvalues):
        # grab our max vector size, short term, and long term memory
        T = self.max_vector
        H = self.short_term_memory
        C = self.long_term_memory
        # our saved inner gate values
        O = self.output_gate
        I = self.input_gate
        C_Tilde = self.update_gate

        # data
        data = self.data
        #activation functions

        sigf = self.Sigf
        sigi = self.Sigi
        sigo = self.Sigo 
        tan1 = self.Tan1
        tan2 = self.Tan2

        #BPTT
        dht  = dvalues[-1,:].reshape(self.n_neurons,1)

        for t in reversed(range(T)):
            # working backwards 
            
            # get data in form we can work with
            xt = data[t].reshape(1,1)
            #backwards prop to get dtan2
            tan2[t].backwards(dht)
            #store output of backwards call
            dtanh2 = tan2[t].dinputs

            # dht with respect to tanh
            dhtdtanh = np.multiply(O[t],dtanh2)

            # dct with respect to dft
            dctdft =np.multiply(dhtdtanh,C[t-1])

            #dct with respect to dit
            dctdit = np.multiply(dhtdtanh,C_Tilde[t])

            #dct with respect to dct_tilde
            dctdct_tilde = np.multiply(dhtdtanh,I[t])

            #backwards prop to get dtan1
            tan1[t].backwards(dctdct_tilde)
            #store output of backwards call
            dtanh1 = tan1[t].dinputs

            # backwards prop of sig for forget gate
            sigf[t].backwards(dctdft)
            dsigmf = sigf[t].dinputs

            # backwards prop of sig for input  gate
            sigi[t].backwards(dctdit)
            dsigmi = sigi[t].dinputs

            # backwards prop of sig for output gate
            sigo[t].backwards(np.multiply(dht,tan2[t].output))
            dsigmo = sigo[t].dinputs

            # derivative to update
            dsigmfdUf = np.dot(dsigmf,xt)
            dsigmfWf = np.dot(dsigmf,H[t-1].T)
            
            #update forget gates weights and bias 
            self.dUf += dsigmfdUf
            self.dwf += dsigmfWf
            self.dbf += dsigmf

            # derivative to update
            dsigmidUi = np.dot(dsigmi,xt)
            dsigmidWi = np.dot(dsigmi,H[t-1].T)
            
            # never drink and drive or you will get a..
            self.dUi +=dsigmidUi
            self.dwi +=dsigmidWi
            self.dbi += dsigmi

            # repeat for output gate
            dsigmodUo = np.dot(dsigmo,xt)
            dsigmodWo = np.dot(dsigmo,H[t-1].T)

            #update weights and bias
            self.dUo +=dsigmodUo
            self.dwo +=dsigmodWo
            self.dbo += dsigmo

            #repeat for dtan
            dtanh1Ug = np.dot(dtanh1,xt)
            dtanh1Wg = np.dot(dtanh1,H[t-1].T)
            
            #update weights and bias                
            self.dUg +=dtanh1Ug
            self.dwg += dtanh1Wg
            self.dbg += dtanh1

            #finally, we find derivative of short term memory
            dht = np.dot(self.wf,dsigmf) + np.dot(self.wi,dsigmi) 
            +np.dot(self.wo,dsigmo) + np.dot(self.wg,dtanh1) 
            +dvalues[t-1,:].reshape(self.n_neurons,1)
        self.short_term_memory = H

class Optimizer_SGD:

    #initializing with a default learning rate of 0.1
    def __init__(self, learning_rate = 1e-5, decay = 0, momentum = 0):
        self.learning_rate  = learning_rate
        self.current_learning_rate = learning_rate
        self.decay  = decay
        self.iterations   = 0
        self.momentum  = momentum
        
    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * \
                (1/ (1 + self.decay*self.iterations))
        
    def update_params(self, layer):
        
        #if we use momentum
        if self.momentum:
            
            #check if layer has attribute "momentum"
            if not hasattr(layer, 'weight_momentums'):
                layer.weight_momentums = np.zeros_like(layer.weights)
                layer.bias_momentums   = np.zeros_like(layer.biases)
                
            #now the momentum parts
            weight_updates = self.momentum * layer.weight_momentums - \
                self.current_learning_rate * layer.dweights
            layer.weight_momentums = weight_updates
            
            bias_updates = self.momentum * layer.bias_momentums - \
                self.current_learning_rate * layer.dbiases
            layer.bias_momentums = bias_updates
            
        else:
            
            weight_updates = -self.current_learning_rate * layer.dweights
            bias_updates   = -self.current_learning_rate * layer.dbias
        
        layer.weights += weight_updates
        layer.bias += bias_updates
        
    def post_update_params(self):
        self.iterations += 1
        

class Optimizer_SGD_LSTM:
    #initializing with a default learning rate of 0.1
    def __init__(self, learning_rate = 1e-5, decay = 0, momentum = 0):
        self.learning_rate  = learning_rate
        self.current_learning_rate = learning_rate
        self.decay  = decay
        self.iterations  = 0
        self.momentum = momentum
        
    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * \
                (1/ (1 + self.decay*self.iterations))
        
    def update_params(self, layer):
        
        #if we use momentum
        if self.momentum:
            
            #check if layer has attribute "momentum"
            if not hasattr(layer, 'Uf_momentums'):
                layer.Uf_momentums = np.zeros_like(layer.Uf)
                layer.Ui_momentums = np.zeros_like(layer.Ui)
                layer.Uo_momentums = np.zeros_like(layer.Uo)
                layer.Ug_momentums = np.zeros_like(layer.Ug)
                
                layer.Wf_momentums = np.zeros_like(layer.Wf)
                layer.Wi_momentums = np.zeros_like(layer.Wi)
                layer.Wo_momentums = np.zeros_like(layer.Wo)
                layer.Wg_momentums = np.zeros_like(layer.Wg)
                
                layer.bf_momentums = np.zeros_like(layer.bf)
                layer.bi_momentums = np.zeros_like(layer.bi)
                layer.bo_momentums = np.zeros_like(layer.bo)
                layer.bg_momentums = np.zeros_like(layer.bg)
                

            #now the momentum parts
            Uf_updates = self.momentum * layer.Uf_momentums - \
                self.current_learning_rate * layer.dUf
            layer.Uf_momentums = Uf_updates
            
            Ui_updates = self.momentum * layer.Ui_momentums - \
                self.current_learning_rate * layer.dUi
            layer.Ui_momentums = Ui_updates
            
            Uo_updates = self.momentum * layer.Uo_momentums - \
                self.current_learning_rate * layer.dUo
            layer.Uo_momentums = Uo_updates
            
            Ug_updates = self.momentum * layer.Ug_momentums - \
                self.current_learning_rate * layer.dUg
            layer.Ug_momentums = Ug_updates
            
            Wf_updates = self.momentum * layer.Wf_momentums - \
                self.current_learning_rate * layer.dWf
            layer.Wf_momentums = Wf_updates
            
            Wi_updates = self.momentum * layer.Wi_momentums - \
                self.current_learning_rate * layer.dWi
            layer.Wi_momentums = Wi_updates
            
            Wo_updates = self.momentum * layer.Wo_momentums - \
                self.current_learning_rate * layer.dWo
            layer.Wo_momentums = Wo_updates
            
            Wg_updates = self.momentum * layer.Wg_momentums - \
                self.current_learning_rate * layer.dWg
            layer.Wg_momentums = Wg_updates
            
            bf_updates = self.momentum * layer.bf_momentums - \
                self.current_learning_rate * layer.dbf
            layer.bf_momentums = bf_updates
            
            bi_updates = self.momentum * layer.bi_momentums - \
                self.current_learning_rate * layer.dbi
            layer.bi_momentums = bi_updates
            
            bo_updates = self.momentum * layer.bo_momentums - \
                self.current_learning_rate * layer.dbo
            layer.bo_momentums = bo_updates
            
            bg_updates = self.momentum * layer.bg_momentums - \
                self.current_learning_rate * layer.dbg
            layer.bg_momentums = bg_updates
            
        else:
            
            Uf_updates = -self.current_learning_rate * layer.dUf
            Ui_updates = -self.current_learning_rate * layer.dUi
            Uo_updates = -self.current_learning_rate * layer.dUo
            Ug_updates = -self.current_learning_rate * layer.dUg
            
            Wf_updates = -self.current_learning_rate * layer.dwf
            Wi_updates = -self.current_learning_rate * layer.dwi
            Wo_updates = -self.current_learning_rate * layer.dwo
            Wg_updates = -self.current_learning_rate * layer.dwg
            
            bf_updates = -self.current_learning_rate * layer.dbf
            bi_updates = -self.current_learning_rate * layer.dbi
            bo_updates = -self.current_learning_rate * layer.dbo
            bg_updates = -self.current_learning_rate * layer.dbg
            
        
        layer.Uf += Uf_updates 
        layer.Ui += Ui_updates 
        layer.Uo += Uo_updates 
        layer.Ug += Ug_updates 
        
        layer.wf += Wf_updates 
        layer.wi += Wi_updates 
        layer.wo += Wo_updates
        layer.wg += Wg_updates
        
        layer.bf += bf_updates 
        layer.bi += bi_updates 
        layer.bo += bo_updates
        layer.bg += bg_updates
        
    def post_update_params(self):
        self.iterations += 1