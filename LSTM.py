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

    def lstm_cell( data,ht,ct,sig_forget_gate,sig_input_gate,sig_output_gate,tan_1,tan_2,H,C,F,O,I,C_tilde ):
        for t, x in enumerate(data):
            x = x.reshape(1,1)