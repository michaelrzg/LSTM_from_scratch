import numpy as np
class LSTM:
    def __init__(self,itterations) -> None:
        self.st_weights = None
        self.input_weights = None
        self.bias = None
        self.short_term=0
        self.long_term=0
        self.itterations = itterations
        # initilize weights and bias
        self.st_weights= [np.random.random() for x in range(4)]
        self.input_weights=  [np.random.random() for x in range(4)]
        self.bias=  [np.random.random() for x in range(4)]
    
    # sigmoid activation funciton
    def sigmoid(self,data):
        return 1/(1 + np.exp(-data))
    
    #tanh activation function
    def tanh(self,data):
        return (np.exp(data)-np.exp(-data))/(np.exp(data)+np.exp(-data))

    def forget_gate(self,data):
        # short term mempry * short term memeory weight + input * input weight + bias for cell
        summation = np.dot([self.short_term,data],[self.st_weights[0],self.input_weights[0]]) + self.bias[0]
        # activation function
        output = self.sigmoid(summation)
        # determine % of long term to remember or 'forget'
        self.long_term = self.long_term*output
        return self.long_term
    
    
    def input_gate(self,data):
        # % to remember from current long term memory
        # short term mempry * short term memeory weight + input * input weight + bias for cell
        summation = ((data * self.input_weights[1]) + self.short_term*self.st_weights[1]) + self.bias[1]
        # activation function
        remember = self.sigmoid(summation)
        
        # potential memory 
        # short term mempry * short term memeory weight + input * input weight + bias for cell
        summation2 = ((data * self.input_weights[2]) + self.short_term*self.st_weights[2]) + self.bias[2]
        # activation function
        potential = self.tanh(summation2)

        self.long_term = self.long_term + (remember*potential)

        return self.long_term

    def update_short_term(self,data):
        # short term mempry * short term memeory weight + input * input weight + bias for cell
        summation = np.dot([self.short_term,data],[self.st_weights[3],self.input_weights[3]]) + self.bias[3]
        # activation function
        remember = self.sigmoid(summation)
        # next get longterm memory contribution
        long_term_contribution = self.tanh(self.long_term)
        #update short term for next run
        self.short_term = remember*long_term_contribution
        return self.short_term
    
    def fit(self,data):
        for i in range(self.itterations):
            for d in data:
                #forget gate
                self.forget_gate(d)
                #input gate
                self.input_gate(d)
                #update short term
                self.update_short_term(d)
        print("Long term: ",self.long_term, " Short term: ", self.short_term)

lstm = LSTM(4)
lstm.fit([0,.5,.25])