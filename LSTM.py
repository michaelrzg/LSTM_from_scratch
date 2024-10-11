import numpy as np

# sigmoid activation function
class Sigmoid_Activation():
    
    # for forward pass usage
    def forward(self, x):
        #calculate sigmoid of x
        sigx = np.clip(1/(1+np.exp(-x)),1e-7,1-1e-1)
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


class LSTM():   
    def __init__(self, n_states, n_features) -> None:
        # num of n_states
        self.n_states = n_states
        # num of features
        self.n_features = n_features
        # learnables
        # forget gate values
        self.Uf = 0.1*np.random.rand(n_states,n_features)
        self.bf = 0.1*np.random.rand(n_states,n_features)
        self.wf = 0.1*np.random.rand(n_states,n_states)

        # input gate values
        self.Ui = 0.1*np.random.rand(n_states,n_features)
        self.bi = 0.1*np.random.rand(n_states,n_features)
        self.wi = 0.1*np.random.rand(n_states,n_states)

        #output gate values
        self.Uo = 0.1*np.random.rand(n_states,n_features)
        self.bo = 0.1*np.random.rand(n_states,n_features)
        self.wo = 0.1*np.random.rand(n_states,n_states)

        #output gate values
        self.Uo = 0.1*np.random.rand(n_states,n_features)
        self.bo = 0.1*np.random.rand(n_states,n_features)
        self.wo = 0.1*np.random.rand(n_states,n_states)

        # c tilde
        self.Ug = 0.1*np.random.rand(n_states,n_features)
        self.bg = 0.1*np.random.rand(n_states,n_features)
        self.wg = 0.1*np.random.rand(n_states,n_states)

    def forward_pass(self,data):
        # find vector length
        self.max_vector = max(data.shape)
        # save data in class
        self.data = data
        # set up lists to store for short term, long term, and c_tilde (last gate) states at timestamps 0 to t
        self.H = [np.zeros((self.n_states,1)) for x in range(self.max_vector+1)]
        self.C  = [np.zeros((self.n_states,1)) for x in range(self.max_vector+1)]
        self.C_tilde    = [np.zeros((self.n_states,1)) for x in range(self.max_vector)]

        # set up lists to store gate inner values at points 0-(t-1) (for debugging later)
        self.F = [np.zeros((self.n_states,1)) for x in range(self.max_vector)]
        self.O  = [np.zeros((self.n_states,1)) for x in range(self.max_vector)]
        self.I = [np.zeros((self.n_states,1)) for x in range(self.max_vector)]    

        self.I = [[np.zeros((self.n_states,1)) for x in range(self.max_vector)]]
        
        #store derivative (gradients) of gate values (delta of learnables)

        # forget gate values
        self.dUf = 0.1*np.random.rand(self.n_states,self.n_features)
        self.dbf = 0.1*np.random.rand(self.n_states,self.n_features)
        self.dwf = 0.1*np.random.rand(self.n_states,self.n_states)

        # input gate values
        self.dUi = 0.1*np.random.rand(self.n_states,self.n_features)
        self.dbi = 0.1*np.random.rand(self.n_states,self.n_features)
        self.dwi = 0.1*np.random.rand(self.n_states,self.n_states)

        #output gate values
        self.dUo = 0.1*np.random.rand(self.n_states,self.n_features)
        self.dbo = 0.1*np.random.rand(self.n_states,self.n_features)
        self.dwo = 0.1*np.random.rand(self.n_states,self.n_states)

        #output gate values
        self.dUo = 0.1*np.random.rand(self.n_states,self.n_features)
        self.dbo = 0.1*np.random.rand(self.n_states,self.n_features)
        self.dwo = 0.1*np.random.rand(self.n_states,self.n_states)

        # c tilde
        self.dUg = 0.1*np.random.rand(self.n_states,self.n_features)
        self.dbg = 0.1*np.random.rand(self.n_states,self.n_features)
        self.dwg = 0.1*np.random.rand(self.n_states,self.n_states)

        # set up activation functions for each gate
        # since each activation function has its own values to store, we need objects
        sig_forget_gate = [Sigmoid_Activation() for x in range(self.max_vector)]
        sig_input_gate = [Sigmoid_Activation() for x in range(self.max_vector)]
        sig_output_gate = [Sigmoid_Activation() for x in range(self.max_vector)]

        #we have 2 tan gates in input and output

        tan_1 = [Tanh_Activation() for x in range(self.max_vector)]
        tan_2 = [Tanh_Activation() for x in range(self.max_vector)]

        # set our initial outer and inner memories at t=0
        ht = self.H[0]
        ct = self.C[0]
        # run cell to progress t and  generate values
        [H, C,sig_forget_gate,sig_input_gate, sig_output_gate, tan_1, tan_2,F,O,I,C_tilde] =self.lstm_cell(
            data,ht,ct,sig_forget_gate,sig_input_gate,sig_output_gate,tan_1,tan_2,self.H,self.C,self.F,self.O,
            self.I,self.C_tilde
        )

        # save outputs of cell
        self.F = F
        self.O = O
        self.I = I
        self.C_tilde = C_tilde
        #save short and long term memory
        self.H=H
        self.C = C
        # save activaton function states
        self.Sigf = sig_forget_gate
        self.Sigi = sig_input_gate
        self.Sigo = sig_output_gate
        self.Tan1 = tan_1
        self.Tan2 = tan_2

    def lstm_cell(self,data,ht,ct,sig_forget_gate,sig_input_gate,sig_output_gate,tan_1,tan_2,H,C,F,O,I,C_tilde ):
        # for each datapoint time stamp
        for t, x in enumerate(data):
            # reshape so we can use dot
            x = x.reshape(1,1)

            # forget gate (determine how much of long term memory we will remember or 'forget')
            # outout = input * weight + short term * short term weight + forget gate bias
            outputf = np.dot(self.Uf, x) + np.dot(self.wf,ht) + self.bf
            # take sigmoid to return value between 0 and 1
            # use our predefined array of sigmoid objects 
            sig_forget_gate[t].forward(outputf)
            #grab output for later
            ft = sig_forget_gate[t].output

            #repeat for input gate
            outputi = np.dot(self.Ui, x) + np.dot(self.wi,ht) + self.bi
            sig_input_gate[t].forward(outputi)
            it = sig_input_gate[t].output

            #repeat for output gate
            outputo = np.dot(self.Uo, x) + np.dot(self.wo,ht) + self.bo
            sig_output_gate[t].forward(outputo)
            ot = sig_output_gate[t].output

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
            return (H,C,sig_forget_gate,sig_input_gate,sig_output_gate,tan_1,tan_2,F,O,I,C_tilde)

    def backward_pass(self,dvalues):
        # grab our max vector size, short term, and long term memory
        T = self.max_vector
        H = self.H
        C = self.C
        # our saved inner gate values
        O = self.O
        I = self.I
        C_Tilde = self.C_tilde

        # data
        data = self.data
        #activation functions

        sigf = self.Sigf
        sigi = self.Sigi
        sigo = self.Sigo 
        tan1 = self.Tan1
        tan2 = self.Tan2

        #BPTT
        dht  = dvalues[-1,:].reshape(self.n_states,1)

        for t in reversed(range(T)):
            # working backwards 
            
            # get data in form we can work with
            xt = data.reshape[t](1,1)

            #backwards prop to get dtan2
            tan2[t].backwards(dht)
            #store output of backwards call
            dtanh2 = tan2[t].dinputs

            # dht with respect to tanh
            dhttanh = np.multiply(O[t],dtanh2)

            # dct with respect to dft
            dctdft =np.multiply(dhttanh,C[t-1])

            #dct with respect to dit
            dctdit = np.multiply(dhttanh,C_Tilde[t])

            #dct with respect to dct_tilde
            dctdct_tilde = np.multiply(dhttanh,I[t])

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
            dsigmfUf = np.dot(dsigmf,xt)
            dsigmfWf = np.dot(dsigmf,H[t-1].T)
            
            #update forget gates weights and bias 
            self.dUf += dsigmfUf
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
            +dvalues[t-1,:].reshape(self.n_states,1)
        self.H = H


