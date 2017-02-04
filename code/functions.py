# some useful functions
import numpy as np
from xman import *

# some useful functions
# declare all operations here first

class f(XManFunctions):
    @staticmethod
    def square(a):
        return XManFunctions.registerDefinedByOperator('square',a)
    @staticmethod
    def sigmoid(A):
        return XManFunctions.registerDefinedByOperator('sigmoid',A)
    @staticmethod
    def tanh(A):
        return XManFunctions.registerDefinedByOperator('tanh',A)
    @staticmethod
    def elem_prod(A,B):
        return XManFunctions.registerDefinedByOperator('elem_prod',A,B)
    @staticmethod
    def relu(A):
        return XManFunctions.registerDefinedByOperator('relu',A)
    @staticmethod
    def softMax(A):
        return XManFunctions.registerDefinedByOperator('softMax',A)
    @staticmethod
    def crossEnt(P, T):
        return XManFunctions.registerDefinedByOperator('crossEnt',P,T)
    @staticmethod
    def mean(a):
        return XManFunctions.registerDefinedByOperator('mean',a)

# the functions that autograd.eval will use to evaluate each function,
# to be called with the functions actual inputs as arguments

EVAL_FUNS = {
    'add':      lambda x1,x2: x1+x2,
    'subtract': lambda x1,x2: x1-x2,
    'square':   np.square,
    'mul': lambda A, B: np.dot(A, B),
    'sigmoid': lambda A: 1.0/(1 + np.exp(-A)),
    'tanh': lambda A: np.tanh(A),
    'elem_prod': lambda A, B: np.multiply(A, B),
    'relu': lambda A: np.maximum(A, 0),
    'softMax': lambda A: softMaxFun(A),
    'crossEnt': lambda P, T: crossEntFun(P, T),
    'mean': lambda a: np.mean(a)
    }

# Takes a matrix and returns a matrix with the same dimensions
def softMaxFun(A):
    normalized = np.exp(A - np.reshape(np.amax(A, axis = 1), (A.shape[0], 1)))
    return normalized*1.0/np.reshape(np.sum(normalized, axis = 1), (normalized.shape[0], 1))

# Takes two matrices and returns a column vector
def crossEntFun(P, T):
    return -1.0*np.sum(np.multiply(T, np.log(P)), axis = 1)
# the functions that autograd.bprop will use in reverse mode
# differentiation.  BP_FUNS[f] is a list of functions df1,....,dfk
# where dfi is used in propagating errors to the i-th input xi of f.
# Specifically, dfi is called with the ordinary inputs to f, with two
# additions: the incoming error, and the output of the function, which
# was computed by autograd.eval in the eval stage.  dfi will return
# delta * df/dxi [f(x1,...,xk)]
# 
# NOTE: Autograd has an optimization where if it finds a softMax op
# followed by crossEnt op, it combines the backward pass for both. So
# you only need to implement the BP_FUNS for the combined operation 
# crossEnt-softMax below.

def _derivAdd(delta,x1):        
    if delta.shape!=x1.shape:
        # broadcast, sum along axis=0
        if delta.shape[1]!=x1.shape[0]:                        
            raise ValueError("Dimension Mismatch")
        return delta.sum(axis=0) #we sum the gradients over the batch
    else: return delta

def reluGradFun(delta,A):
    delta[A <= 0] = 0.0
    return delta
    
def meanGradFun(delta, a):
    N = max(a.shape)
    a = np.array([1.0/N]*N)
    return np.multiply(delta, a)
    
BP_FUNS = {
    'add': [lambda delta,out,x1,x2: _derivAdd(delta,x1),    lambda delta,out,x1,x2: _derivAdd(delta,x2)],
    'subtract': [lambda delta,out,x1,x2: _derivAdd(delta,x1),    lambda delta,out,x1,x2: -_derivAdd(delta,x2)],
    'square': [lambda delta,out,x : delta * 2.0 * x],
    'mul': [lambda delta,out,A,B: np.dot(delta,np.transpose(B)), lambda delta,out,A,B: np.dot(np.transpose(A),delta)],
    'sigmoid': [lambda delta,out,A: np.multiply(delta, np.multiply(out, 1-out))],
    'tanh': [lambda delta,out,A: np.multiply(delta, 1-np.multiply(out,out))],
    'elem_prod': [lambda delta,out,A,B: np.multiply(delta, B), lambda delta,out,A,B: np.multiply(delta, A)],         
    'relu': [lambda delta,out,A: reluGradFun(delta,A)],
    'crossEnt-softMax': [lambda delta,out,O2,T: 1.0*(softMaxFun(O2)-T)/O2.shape[0], lambda delta,out,O2,T: np.zeros(O2.shape)],
    'mean': [lambda delta,out,a: meanGradFun(delta, a)]
    }
