# dnnbuilder

A Python class for configuration-oriented creation of deep neural networks. When used in Python notebooks, graphs of cost function
and accuracy of prediction on training and testing sets are shown and updated in real time during training.

Better documentation with self-contained examples will be added later.

#### Default network parameters:

```python
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
```

#### Adding parameters to define network layout:

```python
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
```    
#### Finally, after creating DataLoaders for training and testing (assume they are named train_loader, test_loader):

```python
trainAcc,testAcc,losses = net.train(train_loader, test_loader)
```

