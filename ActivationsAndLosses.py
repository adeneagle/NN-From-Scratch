import numpy as np

def getActivation(x, activation, derivative=False):
    if activation == "sigmoid":
        if derivative == False:
            return 1.0/(1.0+np.exp(-x))
        else:
            return 1.0/(1.0+np.exp(-x))*(1.0-1.0/(1.0+np.exp(-x)))
    elif activation == "linear":
        if derivative == False:
            return x
        if derivative == True:
            return 1

def getLoss(predicted, ground, loss, derivative=False):
    if loss == "MSE":
        if derivative == False:
            return 1/2*(predicted-ground)**2
        else:
            return (predicted-ground)