"""
Long Short Term Memory for character level entity classification
"""
import sys
import argparse
import numpy as np
from xman import *
from utils import *
from autograd import *
from copy import deepcopy

np.random.seed(0)

class LSTM(object):
    """
    Long Short Term Memory + Feedforward layer
    Accepts maximum length of sequence, input size, number of hidden units and output size
    """
    def __init__(self, max_len, in_size, num_hid, out_size):
        self.M = max_len
        self.V = in_size
        self.h = num_hid
        self.C = out_size
        self.my_xman= self._build() # Store the output of xman.setup() in this variable

    def _build(self):
        x = XMan()
        
        # Model for the LSTM part
        x.H = f.input(name = 'H', default = np.zeros((1, self.h)))
        x.C = f.input(name = 'C', default = np.zeros((1, self.h)))
        C_t = x.C
        H_t = x.H
        
        a = np.sqrt(6.0/(self.V + self.h))
        b = np.sqrt(6.0/(self.h + self.h))
        x.Wi = f.param(name = 'Wi', default = np.random.uniform(-a, a, (self.V, self.h)))
        x.bi = f.param(name = 'bi', default = np.random.uniform(-0.1, 0.1, self.h))
        x.Ui = f.param(name = 'Ui', default = np.random.uniform(-b, b, (self.h, self.h)))
        
        x.Wf = f.param(name = 'Wf', default = np.random.uniform(-a, a, (self.V, self.h)))
        x.bf = f.param(name = 'bf', default = np.random.uniform(-0.1, 0.1, self.h))
        x.Uf = f.param(name = 'Uf', default = np.random.uniform(-b, b, (self.h, self.h)))
        
        x.Wo = f.param(name = 'Wo', default = np.random.uniform(-a, a, (self.V, self.h)))
        x.bo = f.param(name = 'bo', default = np.random.uniform(-0.1, 0.1, self.h))
        x.Uo = f.param(name = 'Uo', default = np.random.uniform(-b, b, (self.h, self.h)))
        
        x.Wc = f.param(name = 'Wc', default = np.random.uniform(-a, a, (self.V, self.h)))
        x.bc = f.param(name = 'bc', default = np.random.uniform(-0.1, 0.1, self.h))
        x.Uc = f.param(name = 'Uc', default = np.random.uniform(-b, b, (self.h, self.h)))
        
        for i in range(self.M):
            x.X = f.input(name = 'X_%d'%(i+1), default=np.random.rand(1, self.V))
            
            temp = f.add(f.mul(x.X, x.Wi), f.mul(H_t, x.Ui))
            I_t = f.sigmoid(f.add(temp, x.bi))
            I_t.name = 'I_%d'%(i+1)
            
            temp = f.add(f.mul(x.X, x.Wf), f.mul(H_t, x.Uf))
            F_t = f.sigmoid(f.add(temp, x.bf))
            F_t.name = 'F_%d'%(i+1)
            
            temp = f.add(f.mul(x.X, x.Wo), f.mul(H_t, x.Uo))
            O_t = f.sigmoid(f.add(temp, x.bo))
            O_t.name = 'O_%d'%(i+1)
            
            temp = f.add(f.mul(x.X, x.Wc), f.mul(H_t, x.Uc))
            C_tilde_t = f.tanh(f.add(temp, x.bc))
            C_tilde_t.name = 'C_tilde_%d'%(i+1)
            
            C_t = f.add(f.elem_prod(F_t, C_t), f.elem_prod(I_t, C_tilde_t))
            H_t = f.elem_prod(O_t, f.tanh(C_t))
            C_t.name = 'C_%d'%(i+1)
            H_t.name = 'H_%d'%(i+1)
        
                
        # Model for the rest of the neural network
        temp1 = np.zeros((1, self.C))
        temp1[0][np.argmax(np.random.uniform(0, 1, self.C))] = 1
        x.T = f.input(name = 'T', default = temp1)
		
        a = np.sqrt(6.0/(self.h + self.C))
        x.W2 = f.param(name = 'W2', default = np.random.uniform(-a, a, (self.h, self.C)))
        x.b2 = f.param(name = 'b2', default = np.random.uniform(-0.1, 0.1, self.C))
        
        x.O2 = f.relu(f.add(f.mul(H_t, x.W2), x.b2))
        x.outputs = f.softMax(x.O2)
        x.loss = f.mean(f.crossEnt(x.outputs, x.T))
        
        return x.setup()

def main(params):
    
    epochs = params['epochs']
    max_len = params['max_len']
    num_hid = params['num_hid']
    batch_size = params['batch_size']
    dataset = params['dataset']
    init_lr = params['init_lr']
    output_file = params['output_file']
    
    # load data and preprocess
    dp = DataPreprocessor()
    data = dp.preprocess('%s.train'%dataset, '%s.valid'%dataset, '%s.test'%dataset)
    # minibatches
    mb_train = MinibatchLoader(data.training, batch_size, max_len, 
           len(data.chardict), len(data.labeldict))
    mb_valid = MinibatchLoader(data.validation, len(data.validation), max_len, 
           len(data.chardict), len(data.labeldict), shuffle=False)
    mb_test = MinibatchLoader(data.test, len(data.test), max_len, 
           len(data.chardict), len(data.labeldict), shuffle=False)
    # build
    print "building lstm..."
    lstm = LSTM(max_len,mb_train.num_chars,num_hid,mb_train.num_labels)
    wengert_list = lstm.my_xman.operationSequence(lstm.my_xman.loss)
    ad = Autograd(lstm.my_xman)
    # train
    print "training..."
    # get default data and params
    value_dict = lstm.my_xman.inputDict()
    lr = init_lr
    best = {}
    min_loss = float('inf')
    for i in range(epochs):
        for (idxs,e,l) in mb_train:
            for j in range(max_len):
                value_dict['X_%d'%(max_len-j)] = e[:, j, :]
            
            value_dict['T'] = l
            value_dict['H'] = np.zeros((e.shape[0], num_hid))
            value_dict['C'] = np.zeros((e.shape[0], num_hid))
            
            value_dict = ad.eval(wengert_list, value_dict)
            gradients = ad.bprop(wengert_list, value_dict, loss = np.float_(1.))
            
            for rname in gradients:
                if lstm.my_xman.isParam(rname):
                    value_dict[rname] = value_dict[rname] - lr*gradients[rname]
                    
        # validate
        for (idxs,e,l) in mb_valid:
            for j in range(max_len):
                value_dict['X_%d'%(max_len-j)] = e[:, j, :]
            
            value_dict['T'] = l
            value_dict['H'] = np.zeros((e.shape[0], num_hid))
            value_dict['C'] = np.zeros((e.shape[0], num_hid))
            
            value_dict = ad.eval(wengert_list, value_dict)
        # Compare current validation loss to minimum validation loss
        # and store params if needed
        if value_dict['loss'] < min_loss:
            min_loss = value_dict['loss']
            for rname in value_dict:
                if lstm.my_xman.isParam(rname):
                    best[rname] = deepcopy(value_dict[rname])
        else:
            lr = 0.5*lr
            
    print "done"
    
    for rname in best:
        value_dict[rname] = best[rname]

    for (idxs,e,l) in mb_test:
        # prepare input and do a fwd pass over it to compute the output probs
        for j in range(max_len):
            value_dict['X_%d'%(max_len-j)] = e[:, j, :]
        value_dict['T'] = l
        value_dict['H'] = np.zeros((e.shape[0], num_hid))
        value_dict['C'] = np.zeros((e.shape[0], num_hid))
        
        value_dict = ad.eval(wengert_list, value_dict)
    
    print value_dict['loss']
    np.save(output_file, value_dict['outputs'])

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_len', dest='max_len', type=int, default=10)
    parser.add_argument('--num_hid', dest='num_hid', type=int, default=50)
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=64)
    parser.add_argument('--dataset', dest='dataset', type=str, default='data/smaller')
    parser.add_argument('--epochs', dest='epochs', type=int, default=15)
    parser.add_argument('--init_lr', dest='init_lr', type=float, default=0.5)
    parser.add_argument('--output_file', dest='output_file', type=str, default='output')
    params = vars(parser.parse_args())
    main(params)
