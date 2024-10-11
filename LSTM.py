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

        self.H=H
        self.C = C

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
            outputf = np.dot(self.Uf, x) + np.dot(self.Wf,ht) + self.bf
            # take sigmoid to return value between 0 and 1
            # use our predefined array of sigmoid objects 
            sig_forget_gate[t].forward(outputf)
            #grab output for later
            ft = sig_forget_gate[t].output

            #repeat for input gate
            outputi = np.dot(self.Ui, x) + np.dot(self.Wi,ht) + self.bi
            sig_input_gate[t].forward(outputi)
            it = sig_input_gate[t].output

            #repeat for output gate
            outputo = np.dot(self.Uo, x) + np.dot(self.Wo,ht) + self.bo
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
            return (H,C,sig_forget_gate,sig_input_gate,sig_output_gate,tan_1,tan_2,F,O,I)




