import numpy as np

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
        # set up lists to store for short term, long term, and c_tilde (last gate) states at points 0-(t-1)
        self.short_term = [np.zeros((self.n_states,1)) for x in range(self.max_vector+1)]
        self.long_term = [np.zeros((self.n_states,1)) for x in range(self.max_vector+1)]
        self.c_tilde = [np.zeros((self.n_states,1)) for x in range(self.max_vector)]

        # set up lists to store gate inner values at points 0-(t-1) (for debugging later)
        self.forget_gate = [np.zeros((self.n_states,1)) for x in range(self.max_vector)]
        self.input_gate = [np.zeros((self.n_states,1)) for x in range(self.max_vector)]
        self.output_gate = [np.zeros((self.n_states,1)) for x in range(self.max_vector)]    
