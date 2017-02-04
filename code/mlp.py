"""
Multilayer Perceptron for character level entity classification
"""
import argparse
import numpy as np
from xman import *
from utils import *
from autograd import *

np.random.seed(0)

class MLP(object):
    """
    Multilayer Perceptron
    Accepts list of layer sizes [in_size, hid_size1, hid_size2, ..., out_size]
    """
    def __init__(self, layer_sizes):
        self.MV = layer_sizes[0]
        self.h = layer_sizes[1]
        self.C = layer_sizes[2]
        self.my_xman = self._build() # Store the output of xman.setup() in this variable
        
        
    def _build(self):
        x = XMan()
        # Define model here
        x.X = f.input(name = 'X', default=np.random.rand(1, self.MV))
        temp = np.zeros((1, self.C))
        temp[0][0] = 1
        x.T = f.input(name = 'T', default=temp)        
        
        a = np.sqrt(6.0/(self.MV + self.h))
        x.W1 = f.param(name = 'W1', default = np.random.uniform(-a, a, (self.MV, self.h)))
        x.b1 = f.param(name = 'b1', default = np.random.uniform(-0.1, 0.1, self.h))
        
        a = np.sqrt(6.0/(self.h + self.C))
        x.W2 = f.param(name = 'W2', default = np.random.uniform(-a, a, (self.h, self.C)))
        x.b2 = f.param(name = 'b2', default = np.random.uniform(-0.1, 0.1, self.C))
        
        x.O1 = f.relu(f.add(f.mul(x.X, x.W1), x.b1))
        x.O2 = f.relu(f.add(f.mul(x.O1, x.W2), x.b2))
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
    print "building mlp..."
    mlp = MLP([max_len*mb_train.num_chars,num_hid,mb_train.num_labels])
    wengert_list = mlp.my_xman.operationSequence(mlp.my_xman.loss)   
    ad = Autograd(mlp.my_xman)
    print "done"
    
    # train
    print "training..."
    # get default data and params
    value_dict = mlp.my_xman.inputDict()
    lr = init_lr
    best = {'W1':0, 'b1':0, 'W2':0, 'b2':0}
    min_loss = float('inf')
    for i in range(epochs):
        for (idxs,e,l) in mb_train:
            # Prepare the input and do a fwd-bckwd pass over it and update the weights
            e = np.reshape(e, (e.shape[0], max_len*mb_train.num_chars))
            value_dict['X'] = e
            value_dict['T'] = l
            
            value_dict = ad.eval(wengert_list, value_dict)
            gradients = ad.bprop(wengert_list, value_dict, loss = np.float_(1.))
            
            for rname in gradients:
                if mlp.my_xman.isParam(rname):
                    if rname == 'b1' or rname == 'b2':
                        value_dict[rname] = value_dict[rname] - lr*np.asarray(gradients[rname]).ravel()
                    else:
                        value_dict[rname] = value_dict[rname] - lr*gradients[rname]
            
        # validate
        for (idxs,e,l) in mb_valid:
            # Prepare the input and do a fwd pass over it to compute the loss
            e = np.reshape(e, (len(data.validation), max_len*mb_valid.num_chars))
            value_dict['X'] = e
            value_dict['T'] = l
            
            value_dict = ad.eval(wengert_list, value_dict)            
        # Compare current validation loss to minimum validation loss
        # and store params if needed
        if value_dict['loss'] < min_loss:
            min_loss = value_dict['loss']
            best['W1'] = value_dict['W1']
            best['b1'] = value_dict['b1']
            best['W2'] = value_dict['W2']
            best['b2'] = value_dict['b2']
        else:
            lr = 0.5*lr
            
    print "done"
    value_dict['W1'] = best['W1']
    value_dict['W2'] = best['W2']
    value_dict['b1'] = best['b1']
    value_dict['b2'] = best['b2']

    for (idxs,e,l) in mb_test:
        # prepare input and do a fwd pass over it to compute the output probs
        e = np.reshape(e, (len(data.test), max_len*mb_test.num_chars))
        value_dict['X'] = e
        value_dict['T'] = l
        
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
