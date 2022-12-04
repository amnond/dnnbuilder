import numpy as np

import torch
import torch.nn as nn
import copy

import json
import datetime
from jupyterplot import ProgressPlot

# --------------------------------------------------------------------
# TODO:use pytorch formulas from docs to take all parameters into account
# Calculate convolution size   
def calc_conv_size(imsize, krnSize=[1,1], padding=[0,0], stride=[1,1] ):
    afterconv_y = np.floor( (imsize[0]+2*padding[0]-krnSize[0]) / stride[0] ) + 1
    afterconv_x = np.floor( (imsize[1]+2*padding[1]-krnSize[1]) / stride[1] ) + 1
    return np.array([int(afterconv_y), int(afterconv_x)])

# Calculate transpose 2d convolution size
def calc_tconv_size(imsize, krnSize=[1,1], padding=[0,0], stride=[1,1] ):
    aftertconv_y = stride[0]*(imsize[0] - 1) + krnSize[0] - 2*padding[0]
    aftertconv_x = stride[1]*(imsize[1] - 1) + krnSize[1] - 2*padding[1]
    return np.array([int(aftertconv_y), int(aftertconv_x)])

# --------------------------------------------------------------------
class LayerWrappers:
    class ToLinear:
        class MakeLinear(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, data):
                return data.view(data.size(0), -1)
            
        def get_torch_layer(self, prev_layer_params, layer_params):
            if 'out2d' in prev_layer_params:
                prev_out2d = prev_layer_params['out2d']
                outsize = prev_out2d[0] * prev_out2d[1] * prev_layer_params['tf']
                layer_params['tn'] = outsize
                
            return self.MakeLinear()
        
    class Conv2d:
        def get_torch_layer(self, prev_layer_params, layer_params):
            layer_name = layer_params['name']
            if not 'tf' in layer_params:
                print(f'Number of out channels not specified in {layer_name}')
                return None
            if not 'out2d' in prev_layer_params:
                prev_name = prev_layer_params['name']
                print(f" Error: output of {prev_name} not compatible as input for {layer_name}" )
                print(f' {str(prev_layer_params)} --> {str(layer_params)}')
                return None
            prev_out_shape = prev_layer_params['out2d']
            krnSize = layer_params['krnsize']
            padding = [0,0]
            if 'padding' in layer_params:
                padding = layer_params['padding']
            stride = [1,1]
            if 'stride' in layer_params:
                padding = layer_params['stride']
            # TODO: add other conv params
            layer_params['out2d'] = calc_conv_size(prev_out_shape, krnSize, padding)
            layer = nn.Conv2d(prev_layer_params['tf'], layer_params['tf'], krnSize, stride, padding)
            if 'no_grad' in layer_params:
                layer.weight.requires_grad = (layer_params['no_grad'] == 1)
            return layer
                    
    class MaxPool2d:
        def get_torch_layer(self, prev_layer_params, layer_params):
            layer_name = layer_params['name']
            if not 'out2d' in prev_layer_params:
                prev_name = prev_layer_params['name']
                print(f" Error: output of {prev_name} not compatible as input for {layer_name}" )
                print(f' {str(prev_layer_params)} --> {str(layer_params)}')
                return None
            prev_out_shape = prev_layer_params['out2d']
            krnSize = layer_params['krnsize']
            
            stride = krnSize
            if 'stride' in layer_params:
                stride = layer_params['stride']
                
            padding = [0,0]
            if 'padding' in layer_params:
                padding = layer_params['padding']
                
            # TODO: add other conv params
            layer_params['out2d'] = calc_conv_size(prev_out_shape, krnSize, padding, stride=stride) 
            layer_params['tf'] = prev_layer_params['tf']
            return nn.MaxPool2d(krnSize)

    class ConvTranspose2d:
        def get_torch_layer(self, prev_layer_params, layer_params):
            layer_name = layer_params['name']
            if not 'tf' in layer_params:
                print(f'Number of out channels not specified in {layer_name}')
                return None
            if not 'out2d' in prev_layer_params:
                prev_name = prev_layer_params['name']
                print(f" Error: output of {prev_name} not compatible as input for {layer_name}" )
                print(f' {str(prev_layer_params)} --> {str(layer_params)}')
                return None
            prev_out_shape = prev_layer_params['out2d']
            krnSize = layer_params['krnsize']
            padding = [0,0]
            if 'padding' in layer_params:
                padding = layer_params['padding']
            stride = [1,1]
            if 'stride' in layer_params:
                padding = layer_params['stride']
            # TODO: add other conv params
            layer_params['out2d'] = calc_tconv_size(prev_out_shape, krnSize, padding)
            return nn.ConvTranspose2d(prev_layer_params['tf'], layer_params['tf'], krnSize, stride, padding)
    
    class Linear:
        def get_torch_layer(self, prev_layer_params, layer_params):
            layer_name = layer_params['name']
            insize = None
            if not 'tn' in prev_layer_params:
                print(f'Number of source neurons not found {layer_name}')
                return None
            
            insize = prev_layer_params['tn']
                
            if not 'tn' in layer_params:
                print(f'Number of target neurons not specified in {layer_name}')
                return None

            layer = nn.Linear(insize, layer_params['tn'])
            if 'no_grad' in layer_params:
                layer.weight.requires_grad = (layer_params['no_grad'] == 1)

            return layer
        
    #------------- Activation functions
    class ReLU:
        def get_torch_layer(self, prev_layer_params, layer_params):
            for attr in ['tn', 'tf', 'out2d']:
                if attr in prev_layer_params:
                    layer_params[attr] = prev_layer_params[attr]
            return nn.ReLU()

    class LeakyReLU:
        def get_torch_layer(self, prev_layer_params, layer_params):
            for attr in ['tn', 'tf', 'out2d']:
                if attr in prev_layer_params:
                    layer_params[attr] = prev_layer_params[attr]
            n_slope = 0.01
            if 'n_slope' in layer_params:
                n_slope = layer_params['n_slope']
            return nn.LeakyReLU(negative_slope=n_slope)

    class Tanh:
        def get_torch_layer(self, prev_layer_params, layer_params):
            for attr in ['tn', 'tf', 'out2d']:
                if attr in prev_layer_params:
                    layer_params[attr] = prev_layer_params[attr]
            return nn.Tanh()
    
    class Sigmoid:
        def get_torch_layer(self, prev_layer_params, layer_params):
            for attr in ['tn', 'tf', 'out2d']:
                if attr in prev_layer_params:
                    layer_params[attr] = prev_layer_params[attr]
            return nn.Sigmoid()

    #------------- Regularization functions
    class Dropout:
        def get_torch_layer(self, prev_layer_params, layer_params):
            for attr in ['tn', 'tf', 'out2d']:
                if attr in prev_layer_params:
                    layer_params[attr] = prev_layer_params[attr]
            dropout_p = 0.5
            if 'p' in layer_params:
                dropout_p = layer_params['p']
            return nn.Dropout(dropout_p)

    class Dropout2d:
        def get_torch_layer(self, prev_layer_params, layer_params):
            for attr in ['tn', 'tf', 'out2d']:
                if attr in prev_layer_params:
                    layer_params[attr] = prev_layer_params[attr]
            dropout_p = 0.5
            if 'p' in layer_params:
                dropout_p = layer_params['p']
            return nn.Dropout2d(dropout_p)

    class BatchNorm1d:
        def get_torch_layer(self, prev_layer_params, layer_params):
            layer_params['tn'] = prev_layer_params['tn']
            return nn.BatchNorm1d(prev_layer_params['tn'])
        
    class BatchNorm2d:
        def get_torch_layer(self, prev_layer_params, layer_params):
            for attr in ['tf', 'out2d']:
                if attr in prev_layer_params:
                    layer_params[attr] = prev_layer_params[attr]
            return nn.BatchNorm2d(prev_layer_params['tf'])
        
# --------------------------------------------------------------------        
class DNN():
    class innerDNN(nn.Module):
        def __init__(self, net_params):
            self.debug = False
            super().__init__()

            layer1_name = 'inputs'
            self.maxlen_layername = len(layer1_name)
            
            input_params = { 'name': layer1_name }
            inputs = net_params['net_input']
            
            if type(inputs) == int:
                input_params['tn'] = inputs
            elif type(inputs) == list:
                ilen = len(inputs)
                if ilen==2:
                    input_params['tf'] = 1
                    input_params['out2d'] = inputs
                elif ilen==3:
                    input_params['tf'] = inputs[2]
                    input_params['out2d'] = [inputs[0], inputs[1]]
            else:
                print(f'Error, invalid network input parameter {str(inputs)}')
                self.ready = False
                return
             
            self.plugin_names = [name for name in dir(LayerWrappers) if name[0]!='_']
            self.layers_params = copy.deepcopy( net_params['layers_params'] )
            nLayers = len(self.layers_params)

            # create dictionary to store the layers
            self.torch_layers = nn.ModuleDict()
            self.nLayers = nLayers       
            
            ### build layers
            self.build_layers(input_params)
                                
            self.ready = True
        
        # --------------------------------------------------------------------
        # forward pass
        def forward(self,x):
            # iterate through layers
            # layers_params - given array of layer types and layer params 
            # plugin_dict - dictionary of layer wrappers derived from layers_params
            # torch_layers - actual PyTorch layers
            mln = self.maxlen_layername
            for layer in self.layers_params:
                layer_name = layer[1]['name']
                out = self.torch_layers[layer_name](x)
                if self.debug:
                    print(f'{layer_name:<{mln}} on {x.shape} => {out.shape}')
                x = out
            return x
        # --------------------------------------------------------------------
        
        def build_layers(self, prev_layer_params):
            self.plugin_dict = {}
            lnum = 0
            for layer in self.layers_params:
                layer_type = layer[0]
                if not layer_type in self.plugin_names:
                    print(f'{layer_type} not in {self.plugin_names}')
                    return False
                    
                layer_plugin = getattr(LayerWrappers, layer_type)
                plugin_inst = layer_plugin()

                if not hasattr(plugin_inst, 'get_torch_layer'):
                    print(f'{layer_type} does not have get_torch_layer method')
                    return False
                
                lnum += 1

                layer_params = layer[1]
                layer_name = f'L{lnum}_{layer_type}'
                if 'name' in layer_params:
                    layer_name = layer_params['name']
                    layer_name = f'L{lnum}_{layer_type}_{layer_name}'

                len_name = len(layer_name)
                if self.maxlen_layername < len_name:
                    self.maxlen_layername = len_name
                
                layer_params['name'] = layer_name
                self.plugin_dict[layer_name] = plugin_inst

                torch_layer = plugin_inst.get_torch_layer(prev_layer_params, layer_params)
                prev_layer_params = layer_params
                if torch_layer == None:
                    return False
                self.torch_layers[layer_name] = torch_layer
            
    
    # DNN class methods
    def __init__(self, net_params):
        
        load_model = False
        if type(net_params)==str:
            filename = net_params
            with open(filename+'.json', 'r') as f:
                net_params = json.load(f)
            load_model = True

        def_params = {
            'optimizer': 'Adam',               # Gradient descent optimization algorithm. Options:
                                               # 'Adadelta','Adagrad','Adam','AdamW','SparseAdam',
                                               # 'Adamax','ASGD','LBFGS','NAdam','RAdam','RMSprop',
                                               # 'Rprop','SGD'
            'lr': 0.0001,                      # learning rate
            'loss_function': 'CrossEntropy',   # loss function used to assess output accuracy. Options:
                                               # 'L1', 'MSE', 'BCE', 'BCEWithLogits', 'NLL', 'PoissonNLL',
                                               # 'CrossEntropy', 'HingeEmbedding', 'MarginRanking',
                                               # 'TripletMargin', 'KLDiv'
            'max_epochs': 30,                  # maximum learning epochs
            'weights_init': 'Kaiming',         # weights initialization method (other option: 'Xavier')
            'use_gpu_if_available': 1,         # 0 always use CPU
            'dropout_rate': 0.25               # percentage of random units per layer whose weight to disregard in training 
        }

        for e in net_params:
            def_params[e] = net_params[e]
        net_params = def_params

        self.device = torch.device('cpu')
        self.devname = ''
        if net_params['use_gpu_if_available']==1 and torch.cuda.is_available():
            self.device = torch.device('cuda:0')
            self.devname = f'({torch.cuda.get_device_name(self.device)})'

        self.net_params = net_params
        
        # create the model instance
        self.net = self.innerDNN(net_params)
            
        self.accuracy_func = None
        
        # create the optimizer
        optifun = getattr( torch.optim, net_params['optimizer'] )
        self.optimizer = optifun(self.net.parameters(),lr=net_params['lr'])

        # create the loss function
        str_loss_func = net_params['loss_function']
        self.lossfun = getattr(nn, str_loss_func +'Loss')()
        
        self.str_loss_func = str_loss_func
        self.accfunc = {}
        self.max_epochs = net_params['max_epochs']

        if str_loss_func == 'BCEWithLogits':
            self.accuracy_func = self.binclassification_accuracy
        
        if str_loss_func == 'CrossEntropy':
            self.accuracy_func = self.classification_accuracy
            
        if str_loss_func == 'MSE':
            self.accuracy_func = self.regression_accuracy

        if load_model:
            self.net.load_state_dict(torch.load(filename+'.pt'))
            self.net.eval()
        elif net_params['weights_init'] == 'Xavier':
            # change the weights (leave biases as Kaiming [default])            
            for p in self.net.named_parameters():
                if 'weight' in p[0]:
                    nn.init.xavier_normal_(p[1].data)         
        
        self.best_model = {'accuracy':0.0, 'net_state':self.net.state_dict()}
        
    def binclassification_accuracy(self, yHat, y):
        predictions = (torch.sigmoid(yHat)>.5).float()        
        return 100*torch.mean((predictions==y).float())
            
    def classification_accuracy(self, yHat, y):
        return 100*torch.mean((torch.argmax(yHat,axis=1)==y).float())
    
    def regression_accuracy(self, yHat, y):
        yh = yHat.detach().numpy().flatten()
        yr = y.detach().numpy().flatten()
        #acc = 100*np.corrcoef(yh,yr)[0,1]
        m1 = np.max([yh,yr])
        m2 = np.min([yh,yr])
        m = m1 - m2
        mdiff = np.mean(np.abs(yh-yr))
        acc = 100-100*mdiff/m
        return acc
        
    def train(self, train_loader, test_loader, max_epochs=None):
        
        trnacc = 'Train Accuracy'
        tstacc = 'Test Accuracy'
        x1 = f"Loss function value ({self.str_loss_func})"
        x2 = ''
        
        pp = ProgressPlot(plot_names=["accuracy", "loss"], line_names=[trnacc,tstacc,x1,x2], y_lim=[[0, 100],[0,1.5]])

        # number of epochs
        prev_loss = 1000
        curr_loss = 0
        numepochs = 0

        # initialize losses
        trainAcc, testAcc, losses  = ([], [], [])

        started_at = datetime.datetime.now()
        
        print( f'Training on {self.device} {self.devname}')
        self.net.to(self.device)
        
        if max_epochs == None:
            max_epochs = self.max_epochs
            
        while (np.abs(prev_loss - curr_loss)) > 0.000001:        

            # loop over training data batches
            batchAcc, batchLoss  = ([], [])

            self.net.train() # Put the network in train mode so dropouts are activated
            
            # iterate over minibatches of training data
            for X,y in train_loader:
                # send data and label tensors to GPU (if exists/requested)
                X = X.to(self.device)
                y = y.to(self.device)
                # forward pass and loss
                yHat = self.net(X)
                loss = self.lossfun(yHat,y)

                # backprop
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # loss from this batch
                batchLoss.append(loss.item())

                # compute accuracy
                batchAcc.append( self.accuracy_func(yHat.cpu(), y.cpu()) )
                # end of batch loop...
                
            # get average batches training accuracy
            trainAcc.append( np.mean(batchAcc) )
            losses.append( np.mean(batchLoss) )
            curr_loss = losses[-1].item()
            
            # test accuracy
            self.net.eval()
            X,y = next(iter(test_loader)) # extract X,y from test dataloader
            X = X.to(self.device)
            y = y.to(self.device)
            
            with torch.no_grad():
                yHat = self.net(X)

            testAcc.append( self.accuracy_func(yHat.cpu(), y.cpu()) )
            last_train_acc = trainAcc[-1].item()
            last_test_acc = testAcc[-1].item()
            
            # Store model if best accuracy so far
            if last_test_acc > self.best_model['accuracy']:
              # new best accuracy
              self.best_model['accuracy'] = last_test_acc

              # model's internal state
              self.best_model['net_state'] = copy.deepcopy( self.net.state_dict() )
            
            numepochs += 1
            
            print(f'Epoch : {numepochs}/{max_epochs}, Test Accuracy:{last_test_acc}', end='\r')
            
            y_update = [[ last_train_acc, last_test_acc, -100, -100], [-10,-10,curr_loss,-10]]
                             
            pp.update(y_update)
            
            if numepochs == max_epochs:
                break
                
        elapsed = datetime.datetime.now() - started_at
        print(f'Completed in {elapsed} seconds after {numepochs} epochs with test accuracy of {testAcc[-1]}')
        pp.finalize()
                
        trainAcc = torch.tensor(trainAcc).cpu()
        testAcc = torch.tensor(testAcc).cpu()
        losses = torch.tensor(losses).cpu()
        
        return trainAcc,testAcc,losses
            
    def save_model(self, to_filename, save_best_model = True):
        net_state = self.best_model['net_state']
        if not save_best_model:
            net_state = self.net.state_dict()
        torch.save(net_state,to_filename+'.pt')
        with open(to_filename + '.json', 'w') as f:
            json.dump(self.net_params, f, indent=4)
        
    def get_model(self):
        return self.net

    def new_instance(self, copy_weights=True):
        newinst = DNN(self.net_params)
        if copy_weights:
            # deepcopy does not cover everything in the torch data representation
            for target,source in zip(newinst.net.named_parameters(),self.net.named_parameters()):
                target[1].data = copy.deepcopy( source[1].data )  
        return newinst

    def test_flow(self):
        inputs = self.net_params['net_input']
        if type(inputs) == int:
            inputs = [inputs]
        li = len(inputs)
        if li == 2 or li == 3:
            datashape = inputs
            inputs = [1] * (4-li)
            inputs.extend(datashape)
        data = torch.tensor(np.random.rand(*inputs)).float()
        self.net.debug = True
        out = self.net(data)        
        self.net.debug = False
        
    def show_params(self):
        for name, param in self.net.named_parameters():
            if param.requires_grad:
                print(name, param.data.shape, param.data)

    def show_netinfo(self):
        print(self.net)
        print(self.optimizer)

    def show_metaparams(self):
        print(json.dumps(self.net_params, indent=4))
        
        
if __name__ == "__main__":

    params = {
        'net_input' : [28,28],
        'layers_params' : [
            [ 'Conv2d',   { 'tf':3, 'krnsize':[5,5], 'padding':[1,1]  } ],
            [ 'MaxPool2d', { 'krnsize': [2,2] } ],
            [ 'BatchNorm2d', {} ],
            [ 'ReLU', {} ],
            [ 'Conv2d',   {'tf':20, 'krnsize':[5,5], 'padding':[1,1] } ],
            [ 'MaxPool2d', { 'krnsize':[2,2] } ],
            [ 'BatchNorm2d', {} ],
            [ 'ReLU', {} ],
            [ 'ToLinear' , {} ],
            [ 'Linear', {'tn':50 } ],
            [ 'ReLU', {} ],            
            [ 'Linear', {'tn':10, 'name':'classifier'},  ]
        ]	
    }
          
    net = DNN(params)
    net.test_flow()
    #print(net.get_model())
    #net.replace_layer('L11_Linear_classifier', [ 'Linear', {'tn':26 },  ])