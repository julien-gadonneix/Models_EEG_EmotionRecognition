import torch
import torch.nn as nn
import torch.nn.functional as F
from models.utils import *
from torch.autograd import Variable



class EEGNet(nn.Module):
    def __init__(self, nb_classes, Chans=64, Samples=128, dropoutRate=0.5, kernLength=64, F1=8, 
                 D=2, F2=16, norm_rate=0.25, nr=1., dropoutType='Dropout'):
        super(EEGNet, self).__init__()
        """ PyTorch Implementation of EEGNet """

        self.name = f'EEGNet-{F1},{D}_kernLength{kernLength}_dropout{dropoutRate}'
        if dropoutType == 'SpatialDropout2D':
            self.dropoutType = nn.Dropout2d
        elif dropoutType == 'Dropout':
            self.dropoutType = nn.Dropout
        else:
            raise ValueError('dropoutType must be one of SpatialDropout2D '
                            'or Dropout, passed as a string.')
        
        self.block1 = nn.Sequential(nn.Conv2d(1, F1, (1, kernLength), padding='same', bias=False),
                                    nn.BatchNorm2d(F1),
                                    ConstrainedConv2d(F1, F1*D, (Chans, 1), bias=False, groups=F1, padding='valid', nr=nr),
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
        self.dense = ConstrainedLinear(F2*int((Samples/4)/8), nb_classes, norm_rate)
    

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.flatten(x)
        x = self.dense(x)
        return F.softmax(x, dim=1)
    


class EEGNet_ChanRed(nn.Module):
    def __init__(self, nb_classes, Chans=64, InnerChans=14, Samples=128, dropoutRate=0.5, kernLength=64, F1=8, 
                 D=2, F2=16, norm_rate=0.25, dropoutType='Dropout'):
        super(EEGNet_ChanRed, self).__init__()
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
        
        self.chan_reduction = nn.Conv2d(1, InnerChans, (Chans, 1), padding='valid')
        
        self.block1 = nn.Sequential(nn.Conv2d(1, F1, (1, kernLength), padding='same', bias=False),
                                    nn.BatchNorm2d(F1),
                                    ConstrainedConv2d(F1, F1*D, (InnerChans, 1), bias=False, groups=F1, padding='valid', nr=1.),
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
        x = self.chan_reduction(x)
        x = torch.permute(x, (0, 2, 1, 3))
        x = self.block1(x)
        x = self.block2(x)
        x = self.flatten(x)
        x = self.dense(x, self.norm_rate)
        return F.softmax(x, dim=1)



# need these for CapsEEGNet
def squash(input_tensor, epsilon=1e-7):
    squared_norm = (input_tensor ** 2 + epsilon).sum(-1, keepdim=True)
    output_tensor = squared_norm * input_tensor / ((1. + squared_norm) * torch.sqrt(squared_norm))
    return output_tensor

class PrimaryCaps(nn.Module):
    def __init__(self, num_capsules=8, in_channels=16, out_channels=32, kernel_size=9, num_routes=32*1*60):
        super(PrimaryCaps, self).__init__()
        self.num_routes = num_routes
        self.capsules = nn.ModuleList([
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, kernel_size), stride=2, padding=0)
            for _ in range(num_capsules)])
        self.relu = nn.ReLU()

    def forward(self, x):
        u = [self.relu(capsule(x)) for capsule in self.capsules]
        u = torch.stack(u, dim=1)
        u = u.view(x.size(0), self.num_routes, -1)
        return squash(u)

class EmotionCaps(nn.Module):
    def __init__(self, num_capsules=3, num_routes=32*1*60, in_channels=8, out_channels=16):
        super(EmotionCaps, self).__init__()
        self.in_channels = in_channels
        self.num_routes = num_routes
        self.num_capsules = num_capsules
        self.W = nn.Parameter(torch.randn(1, num_routes, num_capsules, out_channels, in_channels))

    def forward(self, x):
        batch_size = x.size(0)
        x = torch.stack([x] * self.num_capsules, dim=2).unsqueeze(4)
        W = torch.cat([self.W] * batch_size, dim=0)
        u_hat = torch.matmul(W, x)
        b_ij = Variable(torch.zeros(1, self.num_routes, self.num_capsules, 1))
        b_ij = b_ij.to(x.device)
        num_iterations = 3
        for iteration in range(num_iterations):
            c_ij = F.softmax(b_ij, dim=1)
            c_ij = torch.cat([c_ij] * batch_size, dim=0).unsqueeze(4)
            s_j = (c_ij * u_hat).sum(dim=1, keepdim=True)
            v_j = squash(s_j)
            if iteration < num_iterations - 1:
                a_ij = torch.matmul(u_hat.transpose(3, 4), torch.cat([v_j] * self.num_routes, dim=1))
                b_ij = b_ij + a_ij.squeeze(4).mean(dim=0, keepdim=True)
        return v_j.squeeze(1)
    


class CapsEEGNet(nn.Module):
    def __init__(self, nb_classes, Chans=14, dropoutRate=0.5, kernLength=64, F1=8, 
                 D=2, F2=32, norm_rate=0.25, dropoutType='Dropout',
                 num_capsules=8, num_routes=32*1*60, kern_size=9, out_dim=16):
        super(CapsEEGNet, self).__init__()
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
        
        self.block_1_2 = nn.Sequential(nn.Conv2d(1, F1, (1, kernLength), padding='same', bias=False),
                                    nn.BatchNorm2d(F1),
                                    nn.ELU(), # added but not present in the other implementation
                                    ConstrainedConv2d(F1, F1*D, (Chans, 1), bias=False, groups=F1, padding='valid', nr=1.),
                                    nn.BatchNorm2d(D*F1),
                                    nn.ELU(),
                                    # nn.AvgPool2d((1, 4)), # present in the other implementation
                                    self.dropoutType(dropoutRate))

        self.primaryCaps = PrimaryCaps(num_capsules=num_capsules, in_channels=F1*D, out_channels=F2, kernel_size=kern_size, num_routes=num_routes)
        self.emotionCaps = EmotionCaps(num_capsules=nb_classes, num_routes=num_routes, in_channels=num_capsules, out_channels=out_dim)
        self.fc = nn.Linear(out_dim, 1)
    

    def forward(self, x):
        x = self.block_1_2(x)
        x = self.primaryCaps(x)
        x = self.emotionCaps(x)
        x = x.squeeze(-1)
        x = self.fc(x)
        x = x.squeeze(-1)
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


