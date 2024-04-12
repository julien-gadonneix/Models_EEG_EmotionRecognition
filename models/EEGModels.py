"""
 ARL_EEGModels - A collection of Convolutional Neural Network models for EEG
 Signal Processing and Classification, using Keras and Tensorflow

 Requirements:
    (1) tensorflow == 2.X (as of this writing, 2.0 - 2.3 have been verified
        as working)
 
 To run the EEG/MEG ERP classification sample script, you will also need

    (4) mne >= 0.17.1
    (5) PyRiemann >= 0.2.5
    (6) scikit-learn >= 0.20.1
    (7) matplotlib >= 2.2.3
    
 To use:
    
    (1) Place this file in the PYTHONPATH variable in your IDE (i.e.: Spyder)
    (2) Import the model as
        
        from EEGModels import EEGNet    
        
        model = EEGNet(nb_classes = ..., Chans = ..., Samples = ...)
        
    (3) Then compile and fit the model
    
        model.compile(loss = ..., optimizer = ..., metrics = ...)
        fitted    = model.fit(...)
        predicted = model.predict(...)

 Portions of this project are works of the United States Government and are not
 subject to domestic copyright protection under 17 USC Sec. 105.  Those 
 portions are released world-wide under the terms of the Creative Commons Zero 
 1.0 (CC0) license.  
 
 Other portions of this project are subject to domestic copyright protection 
 under 17 USC Sec. 105.  Those portions are licensed under the Apache 2.0 
 license.  The complete text of the license governing this material is in 
 the file labeled LICENSE.TXT that is a part of this project's official 
 distribution. 
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.utils import *



class EEGNet(nn.Module):
    def __init__(self, nb_classes, Chans=64, Samples=128, dropoutRate=0.5, kernLength=64, F1=8, 
                 D=2, F2=16, norm_rate=0.25, dropoutType='Dropout'):
        super(EEGNet, self).__init__()
        """ PyTorch Implementation of EEGNet """

        self.name = f'EEGNet-{F1},{D}_kernLength{kernLength}_dropout{dropoutRate}'
        self.norm_rate = norm_rate
        if dropoutType == 'SpatialDropout2D':
            self.dropoutType = nn.Dropout2d
        elif dropoutType == 'Dropout':
            self.dropoutType = nn.Dropout
        else:
            raise ValueError('dropoutType must be one of SpatialDropout2D '
                            'or Dropout, passed as a string.')
        
        self.block1 = nn.Sequential(nn.Conv2d(1, F1, (1, kernLength), padding='same', bias=False),
                                    nn.BatchNorm2d(F1),
                                    ConstrainedConv2d(F1, F1*D, (Chans, 1), bias=False, groups=F1, padding='valid', nr=1.),
                                    nn.BatchNorm2d(D*F1),
                                    nn.ELU(),
                                    nn.AvgPool2d((1, 4)),
                                    self.dropoutType(dropoutRate))
        self.block2 = nn.Sequential(SeparableConv2d(F1*D, F2, (1, 16), padding='same', bias=False),
                                    nn.BatchNorm2d(F2),
                                    nn.ELU(),
                                    nn.AvgPool2d((1, 8)),
                                    self.dropoutType(dropoutRate))
        self.flatten = nn.Flatten()
        self.dense = ConstrainedLinear(F2*int((Samples/4)/8), nb_classes)
    

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.flatten(x)
        x = self.dense(x, self.norm_rate)
        return F.softmax(x, dim=1)
    


class EEGNet_SSVEP(nn.Module):
    def __init__(self, nb_classes, Chans=64, Samples=128, dropoutRate=0.5, kernLength=64, F1=8, 
                 D=2, F2=16, norm_rate=0.25, dropoutType='Dropout'):
        super(EEGNet_SSVEP, self).__init__()
        """ PyTorch Implementation of SSVEP Variant of EEGNet """

        self.name = f'EEGNet_SSVEP-{F1},{D}_kernLength{kernLength}_dropout{dropoutRate}'
        self.norm_rate = norm_rate
        if dropoutType == 'SpatialDropout2D':
            self.dropoutType = nn.Dropout2d
        elif dropoutType == 'Dropout':
            self.dropoutType = nn.Dropout
        else:
            raise ValueError('dropoutType must be one of SpatialDropout2D '
                            'or Dropout, passed as a string.')
        
        self.block1 = nn.Sequential(nn.Conv2d(1, F1, (1, kernLength), padding='same', bias=False),
                                    nn.BatchNorm2d(F1),
                                    ConstrainedConv2d(F1, F1*D, (Chans, 1), bias=False, groups=F1, padding='valid', nr=1.),
                                    nn.BatchNorm2d(D*F1),
                                    nn.ELU(),
                                    nn.AvgPool2d((1, 4)),
                                    self.dropoutType(dropoutRate))
        self.block2 = nn.Sequential(SeparableConv2d(F1*D, F2, (1, 16), padding='same', bias=False),
                                    nn.BatchNorm2d(F2),
                                    nn.ELU(),
                                    nn.AvgPool2d((1, 8)),
                                    self.dropoutType(dropoutRate))
        self.flatten = nn.Flatten()
        self.dense = nn.Linear(F2*int((Samples/4)/8), nb_classes)
    

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.flatten(x)
        x = self.dense(x)
        return F.softmax(x, dim=1)



class DeepConvNet(nn.Module):
    def __init__(self, nb_classes, Chans=64, Samples=256, dropoutRate=0.5):
        super(DeepConvNet, self).__init__()
        """ PyTorch Implementation of Deep Convolutional Network """

        self.name = 'DeepConvNet'
        self.block1 = nn.Sequential(ConstrainedConv2d(1, 25, (1, 5), padding='valid', nr=2.),
                                    ConstrainedConv2d(25, 25, (Chans, 1), nr=2.),
                                    nn.BatchNorm2d(25),
                                    nn.ELU(),
                                    nn.MaxPool2d((1, 2), (1, 2)),
                                    nn.Dropout(dropoutRate))
        self.block2 = nn.Sequential(ConstrainedConv2d(25, 50, (1, 5), padding='valid', nr=2.),
                                    nn.BatchNorm2d(50),
                                    nn.ELU(),
                                    nn.MaxPool2d((1, 2), (1, 2)),
                                    nn.Dropout(dropoutRate))
        self.block3 = nn.Sequential(ConstrainedConv2d(50, 100, (1, 5), padding='valid', nr=2.),
                                    nn.BatchNorm2d(100),
                                    nn.ELU(),
                                    nn.MaxPool2d((1, 2), (1, 2)),
                                    nn.Dropout(dropoutRate))
        self.block4 = nn.Sequential(ConstrainedConv2d(100, 200, (1, 5), padding='valid', nr=2.),
                                    nn.BatchNorm2d(200),
                                    nn.ELU(),
                                    nn.MaxPool2d((1, 2), (1, 2)),
                                    nn.Dropout(dropoutRate))
        self.flatten = nn.Flatten()
        self.dense = ConstrainedLinear(200*int((((Samples/2)/2)/2)/2), nb_classes)
    

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.flatten(x)
        x = self.dense(x, 0.5)



# need these for ShallowConvNet
def square(x):
    return torch.square(x)

def log(x):
    return torch.log(torch.clamp(x, min=1e-7, max=10000)) 


class ShallowConvNet(nn.Module):
    def __init__(self, nb_classes, Chans=64, Samples=128, dropoutRate=0.5):
        super(ShallowConvNet, self).__init__()
        """ PyTorch Implementation of Shallow Convolutional Network """

        self.name = 'ShallowConvNet'
        self.block1 = nn.Sequential(ConstrainedConv2d(1, 40, (1, 13), padding='same', nr=2.),
                                    ConstrainedConv2d(40, 40, (Chans, 1), bias=False, nr=2.),
                                    nn.BatchNorm2d(40),
                                    nn.ELU(),
                                    square,
                                    nn.AvgPool2d((1, 35), (1, 7)),
                                    log)
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(dropoutRate)
        self.dense = ConstrainedLinear(40*int((Samples/7)/35), nb_classes)
    

    def forward(self, x):
        x = self.block1(x)
        x = self.flatten(x)
        x = self.dense(x, 0.5)
        return F.softmax(x, dim=1)




# def EEGNet_old(nb_classes, Chans = 64, Samples = 128, regRate = 0.0001,
#            dropoutRate = 0.25, kernels = [(2, 32), (8, 4)], strides = (2, 4)):
#     """ Keras Implementation of EEGNet_v1 (https://arxiv.org/abs/1611.08024v2)

#     This model is the original EEGNet model proposed on arxiv
#             https://arxiv.org/abs/1611.08024v2
    
#     with a few modifications: we use striding instead of max-pooling as this 
#     helped slightly in classification performance while also providing a 
#     computational speed-up. 
    
#     Note that we no longer recommend the use of this architecture, as the new
#     version of EEGNet performs much better overall and has nicer properties.
    
#     Inputs:
        
#         nb_classes     : total number of final categories
#         Chans, Samples : number of EEG channels and samples, respectively
#         regRate        : regularization rate for L1 and L2 regularizations
#         dropoutRate    : dropout fraction
#         kernels        : the 2nd and 3rd layer kernel dimensions (default is 
#                          the [2, 32] x [8, 4] configuration)
#         strides        : the stride size (note that this replaces the max-pool
#                          used in the original paper)
    
#     """

#     # start the model
#     input_main   = Input((Chans, Samples))
#     layer1       = Conv2D(16, (Chans, 1), input_shape=(Chans, Samples, 1),
#                                  kernel_regularizer = l1_l2(l1=regRate, l2=regRate))(input_main)
#     layer1       = BatchNormalization()(layer1)
#     layer1       = Activation('elu')(layer1)
#     layer1       = Dropout(dropoutRate)(layer1)
    
#     permute_dims = 2, 1, 3
#     permute1     = Permute(permute_dims)(layer1)
    
#     layer2       = Conv2D(4, kernels[0], padding = 'same', 
#                             kernel_regularizer=l1_l2(l1=0.0, l2=regRate),
#                             strides = strides)(permute1)
#     layer2       = BatchNormalization()(layer2)
#     layer2       = Activation('elu')(layer2)
#     layer2       = Dropout(dropoutRate)(layer2)
    
#     layer3       = Conv2D(4, kernels[1], padding = 'same',
#                             kernel_regularizer=l1_l2(l1=0.0, l2=regRate),
#                             strides = strides)(layer2)
#     layer3       = BatchNormalization()(layer3)
#     layer3       = Activation('elu')(layer3)
#     layer3       = Dropout(dropoutRate)(layer3)
    
#     flatten      = Flatten(name = 'flatten')(layer3)
    
#     dense        = Dense(nb_classes, name = 'dense')(flatten)
#     softmax      = Activation('softmax', name = 'softmax')(dense)
    
#     return Model(inputs=input_main, outputs=softmax)


