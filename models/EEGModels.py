import torch
import torch.nn as nn
import torch.nn.functional as F
from models.utils import *
from torch.autograd import Variable
import math
from einops.layers.torch import Rearrange
from einops import rearrange
import numpy as np



class EEGNet(nn.Module):
    def __init__(self, nb_classes, Chans=64, Samples=128, dropoutRate=0.5, kernLength=64, F1=8, 
                 D=2, F2=16, norm_rate=0.25, nr=1., dropoutType='Dropout'):
        super(EEGNet, self).__init__()
        """ PyTorch Implementation of EEGNet """

        self.name = 'EEGNet'
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
                 D=2, F2=16, norm_rate=0.25, nr=1., dropoutType='Dropout'):
        super(EEGNet_ChanRed, self).__init__()
        """ PyTorch Implementation of Caps-EEGNet """

        self.name = 'EEGNet'
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
                                    ConstrainedConv2d(F1, F1*D, (InnerChans, 1), bias=False, groups=F1, padding='valid', nr=nr),
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
        x = self.chan_reduction(x)
        x = torch.permute(x, (0, 2, 1, 3))
        x = self.block1(x)
        x = self.block2(x)
        x = self.flatten(x)
        x = self.dense(x)
        return F.softmax(x, dim=1)
    


class EEGNet_WT(nn.Module):
    def __init__(self, nb_classes, Chans=64, InnerChans=14, Samples=128, dropoutRate=0.5, kernLength=64, F1=8, 
                 D=2, F2=16, norm_rate=0.25, nr=1., dropoutType='Dropout', nb_freqs=48):
        super(EEGNet_WT, self).__init__()
        """ PyTorch Implementation of EEGNet zith wavelt transform as input """

        self.name = 'EEGNet'
        self.nb_freqs = nb_freqs
        self.InnerChans = InnerChans
        if dropoutType == 'SpatialDropout2D':
            self.dropoutType = nn.Dropout2d
        elif dropoutType == 'Dropout':
            self.dropoutType = nn.Dropout
        else:
            raise ValueError('dropoutType must be one of SpatialDropout2D '
                            'or Dropout, passed as a string.')
        
        self.chan_reduction = nn.Conv2d(nb_freqs, nb_freqs*InnerChans, (Chans, 1), groups=nb_freqs, padding='valid')
        
        self.block1 = nn.Sequential(nn.Conv2d(nb_freqs, F1, (1, kernLength), padding='same', bias=False),
                                    nn.BatchNorm2d(F1),
                                    ConstrainedConv2d(F1, F1*D, (InnerChans, 1), bias=False, groups=F1, padding='valid', nr=nr),
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
        if self.nb_freqs > 1:
            x = torch.permute(x, (0, 2, 1, 3))
        x = self.chan_reduction(x)
        x = x.view(x.shape[0], self.nb_freqs, self.InnerChans, x.shape[3])
        x = self.block1(x)
        x = self.block2(x)
        x = self.flatten(x)
        x = self.dense(x)
        return F.softmax(x, dim=1)



# need these for CapsEEGNet
def squash(input_tensor, epsilon=1e-7):
    squared_norm = (input_tensor ** 2 + epsilon).sum(-1, keepdim=True)
    output_tensor = squared_norm * input_tensor / ((1. + squared_norm) * torch.sqrt(squared_norm))
    return output_tensor

class PrimaryCap(nn.Module):
    '''
    v0: no concatenation: performs poorly
    v1: concatenate with input: difficult ot use because of the different dimensions
    v2: concatenate with input and apply 1x1 convolution: performs well
    '''
    def __init__(self, inputs, dim_capsule, n_channels, kernel_size, strides, padding, model_version, dev):
        super(PrimaryCap, self).__init__()
        self.model_version = model_version
        self.dim_capsule = dim_capsule
        self.caps = nn.Conv2d(inputs, dim_capsule*n_channels, kernel_size=kernel_size, stride=strides, padding=padding, device=dev)
        if model_version == 'v2':
            self.caps2 = nn.Conv2d((dim_capsule*n_channels)+inputs, dim_capsule*n_channels, kernel_size=1, stride=1, padding='valid', device=dev)
    
    def forward(self, x):
        out = self.caps(x)
        if self.model_version != 'v0':
            out = torch.cat([x, out], dim=1)
        if self.model_version == 'v2':
            out = self.caps2(out)
        out = out.reshape(out.shape[0], -1, self.dim_capsule)
        return squash(out)

class EmotionCap(nn.Module):
    def __init__(self, num_capsule, dim_capsule, routings, input_num_capsule, input_dim_capsule, dev):
        super(EmotionCap, self).__init__()
        self.num_capsule = num_capsule
        self.dim_capsule = dim_capsule
        self.routings = routings
        self.input_num_capsule = input_num_capsule
        self.input_dim_capsule = input_dim_capsule
        self.W = nn.Parameter(nn.init.xavier_uniform_(torch.zeros(self.num_capsule, self.input_num_capsule, self.dim_capsule, self.input_dim_capsule, device=dev)))

    def forward(self, x):
        u_expand = torch.unsqueeze(x, 1)
        u_tiled = torch.tile(u_expand, (1, self.num_capsule, 1, 1))
        u_hat = torch.matmul(self.W, u_tiled.unsqueeze(-1)).squeeze(-1)
        b = Variable(torch.zeros((u_hat.shape[0], self.num_capsule, 1, self.input_num_capsule), device=x.device))
        for i in range(self.routings):
            c = F.softmax(b, dim=1)
            s = torch.matmul(c, u_hat)
            v = squash(s).permute(0, 1, 3, 2)
            if i < self.routings - 1:
                a = torch.matmul(u_hat, v)
                b += a.permute(0, 1, 3, 2)
        return v.squeeze(-1)

class PrimaryCaps(nn.Module):
    def __init__(self, num_capsules, in_channels, out_channels, kernel_size, num_routes):
        super(PrimaryCaps, self).__init__()
        self.num_routes = num_routes
        self.capsules = nn.ModuleList([
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=2, padding=0)
            for _ in range(num_capsules)])
        self.relu = nn.ReLU()

    def forward(self, x):
        u = [self.relu(capsule(x)) for capsule in self.capsules]
        u = torch.stack(u, dim=1)
        u = u.view(x.size(0), self.num_routes, -1)
        return squash(u)

class EmotionCaps(nn.Module):
    def __init__(self, num_capsules, num_routes, in_channels, out_channels):
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
    def __init__(self, nb_classes, Chans):
        super(CapsEEGNet, self).__init__()
        """ PyTorch Implementation of Caps-EEGNet """

        self.name = 'CapsEEGNet'
        self.dropoutType = nn.Dropout
        
        self.block_1_2 = nn.Sequential(nn.Conv2d(1, 8, (1, 64), padding='same', bias=False),
                                    nn.BatchNorm2d(8),
                                    nn.ELU(), # added but not present in EEGNet
                                    ConstrainedConv2d(8, 8*2, (Chans, 1), bias=False, groups=8, padding='valid', nr=1.),
                                    nn.BatchNorm2d(8*2),
                                    nn.ELU(),
                                    # nn.AvgPool2d((1, 4)), # present in EEGNet
                                    self.dropoutType(0.5))

        # self.primaryCaps = PrimaryCaps(num_capsules=8, in_channels=8*2, out_channels=32, kernel_size=(1, 6), num_routes=32*1*60)
        # self.emotionCaps = EmotionCaps(num_capsules=nb_classes, num_routes=32*1*60, in_channels=8, out_channels=16)
        self.primaryCaps = PrimaryCap(8*2, 8, 32, (1, 6), 1, 'same', 'v2', None) # ReLU activation after the convolutional layer not present + residual
        self.emotionCaps = EmotionCap(nb_classes, 16, 3, 32*1*128, 8, None)
        self.fc = nn.Linear(16, 1)
    

    def forward(self, x):
        x = self.block_1_2(x)
        x = self.primaryCaps(x)
        x = self.emotionCaps(x)
        x = x.squeeze(-1)
        x = self.fc(x)
        x = x.squeeze(-1)
        return F.softmax(x, dim=1)
    


# need these for TCNet
class ShiftedWindowMSA(nn.Module):
    def __init__(self, emb_size, num_heads, dev, window_size=2, shifted=True):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.window_size = window_size
        self.shifted = shifted
        self.linear1 = nn.Linear(emb_size, 3*emb_size, device=dev)
        self.linear2 = nn.Linear(emb_size, emb_size, device=dev)
        self.pos_embeddings = nn.Parameter(torch.randn(window_size*2 - 1, window_size*2 - 1, device=dev))
        self.indices = torch.tensor(np.array([[x, y] for x in range(window_size) for y in range(window_size)]), device=dev)
        self.relative_indices = self.indices[None, :, :] - self.indices[:, None, :]
        self.relative_indices += self.window_size - 1

    def forward(self, x):
        h_dim = self.emb_size / self.num_heads
        _, _, height, width = x.shape
        x = rearrange(x, 'b c h w -> b (h w) c')
        x = self.linear1(x)
        x = rearrange(x, 'b (h w) (c k) -> b h w c k', h=height, w=width, k=3, c=self.emb_size)
        if self.shifted:
            x = torch.roll(x, (-self.window_size//2, -self.window_size//2), dims=(1,2))
        x = rearrange(x, 'b (Wh w1) (Ww w2) (e H) k -> b H Wh Ww (w1 w2) e k', w1=self.window_size, w2=self.window_size, H=self.num_heads)            
        Q, K, V = x.chunk(3, dim=6)
        Q, K, V = Q.squeeze(-1), K.squeeze(-1), V.squeeze(-1)
        wei = (Q @ K.transpose(4,5)) / np.sqrt(h_dim)
        rel_pos_embedding = self.pos_embeddings[self.relative_indices[:, :, 0], self.relative_indices[:, :, 1]]
        wei += rel_pos_embedding
        if self.shifted:
            row_mask = torch.zeros((self.window_size**2, self.window_size**2), device=x.device)
            row_mask[-self.window_size * (self.window_size//2):, 0:-self.window_size * (self.window_size//2)] = float('-inf')
            row_mask[0:-self.window_size * (self.window_size//2), -self.window_size * (self.window_size//2):] = float('-inf')
            column_mask = rearrange(row_mask, '(r w1) (c w2) -> (w1 r) (w2 c)', w1=self.window_size, w2=self.window_size)
            wei[:, :, -1, :] += row_mask
            wei[:, :, :, -1] += column_mask
        wei = F.softmax(wei, dim=-1) @ V
        x = rearrange(wei, 'b H Wh Ww (w1 w2) e -> b (Wh w1) (Ww w2) (H e)', w1=self.window_size, w2=self.window_size, H=self.num_heads)
        x = rearrange(x, 'b h w c -> b (h w) c')
        return self.linear2(x)
    
class MLP(nn.Module):
    def __init__(self, emb_size, dev):
        super().__init__()
        self.ff = nn.Sequential(
                         nn.Linear(emb_size, 4*emb_size, device=dev),
                         nn.GELU(),
                         nn.Linear(4*emb_size, emb_size, device=dev)
                  )
    
    def forward(self, x):
        return self.ff(x)
    
class SwinEncoder(nn.Module):
    def __init__(self, emb_size, num_heads, dev, window_size=2):
        super().__init__()
        self.WMSA = ShiftedWindowMSA(emb_size, num_heads, dev, window_size, shifted=False)
        self.SWMSA = ShiftedWindowMSA(emb_size, num_heads, dev, window_size, shifted=True)
        self.ln = nn.LayerNorm(emb_size, device=dev)
        self.MLP = MLP(emb_size, dev)
        
    def forward(self, x):
        _, _, height, width = x.shape
        x_reshaped = rearrange(x, 'b c h w -> b (h w) c')
        x_ln = self.ln(x_reshaped)
        x = rearrange(x_ln, 'b (h w) c -> b c h w', h=height, w=width)
        x = x_reshaped + self.WMSA(x)
        x = x + self.MLP(self.ln(x))

        x_ln = self.ln(x)
        x_reshaped =  rearrange(x_ln, 'b (h w) c -> b c h w', h=height, w=width)
        x = x + self.SWMSA(x_reshaped)
        x = x + self.MLP(self.ln(x))

        return x

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=200):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
        return self.dropout(x)
    


class TCNet(nn.Module):
    def __init__(self, nb_classes, Chans, dev0=None, dev1=None):
        super(TCNet, self).__init__()
        """ PyTorch Implementation of TC-Net """

        self.name = 'TCNet'
        self.dev0 = dev0
        self.dev1 = dev1
        
        d = 32
        self.PatchPartition = nn.Conv2d(Chans, d, (3, 4), stride=(3, 4), device=dev0)
        # self.EEG_Transformer = nn.ModuleList()
        # self.PositionalEncoding = nn.ModuleList()
        self.stages = nn.ModuleList()
        self.PatchMerging = nn.ModuleList()
        for i in range(4):
            # self.PositionalEncoding.append(PositionalEncoding(d*4, max_len=int(128/(4**i))))
            # encoder_layer = nn.TransformerEncoderLayer(d_model=d*4, nhead=1, batch_first=True, norm_first=True, device=device)
            # self.EEG_Transformer.append(nn.TransformerEncoder(encoder_layer, num_layers=1, enable_nested_tensor=False))
            # # self.EEG_Transformer.append(nn.Transformer(d_model=d*4, batch_first=True, norm_first=True))
            self.stages.append(SwinEncoder(d, 1, dev0, window_size=2))
            if i < 3:
                self.PatchMerging.append(nn.Conv2d(d, d*2, (4, 4), stride=(2, 2), padding=(1, 1), device=dev0))
                d *= 2
        # self.primaryCaps = PrimaryCaps(num_capsules=d, in_channels=8, out_channels=8, kernel_size=6, num_routes=8*1*124)
        # self.emotionCaps = EmotionCaps(num_capsules=nb_classes, num_routes=8*1*124, in_channels=d, out_channels=16)
        self.primaryCaps = PrimaryCap(d, 8, 48, 6, 1, 'same', 'v2', dev1)
        self.emotionCaps = EmotionCap(nb_classes, 16, 3, 8*48, 8, dev1)
    

    def forward(self, x):
        x = self.PatchPartition(x)
        # bs, fs, hs, ws = x.shape
        for i in range(len(self.stages)):
            _, _, height, width = x.shape
            # x = x.view(bs, fs*(2**i)*4, -1)
            # x = x.permute(0, 2, 1)
            # x = self.PositionalEncoding[i](x) * math.sqrt(fs*(2**i)*4)
            # x = self.EEG_Transformer[i](x)
            x = self.stages[i](x)
            x = rearrange(x, 'b (h w) c -> b c h w', h=height, w=width)
            # x = x.permute(0, 2, 1)
            # x = x.reshape(bs, fs*(2**i), hs//(2**i), ws//(2**i))
            if i < len(self.PatchMerging):
                x = self.PatchMerging[i](x)
        if self.dev1 is not None:
            x = x.to(self.dev1)
        x = self.primaryCaps(x)
        x = self.emotionCaps(x)
        return torch.norm(x, dim=2).squeeze()



class TCNet_EMD(nn.Module):
    def __init__(self, nb_classes, Chans, nb_freqs, kern_emd, dev0=None, dev1=None):
        super(TCNet_EMD, self).__init__()
        """ PyTorch Implementation of TC-Net """

        self.name = 'TCNet'
        self.dev0 = dev0
        self.dev1 = dev1

        self.emd_expansion = nn.Conv2d(nb_freqs+1, 48, (1, kern_emd), padding='same', device=dev0)
        
        d = 32
        self.PatchPartition = nn.Conv2d(Chans, d, (3, 4), stride=(3, 4), device=dev0)
        # self.EEG_Transformer = nn.ModuleList()
        # self.PositionalEncoding = nn.ModuleList()
        self.stages = nn.ModuleList()
        self.PatchMerging = nn.ModuleList()
        for i in range(4):
            # self.PositionalEncoding.append(PositionalEncoding(d*4, max_len=int(128/(4**i))))
            # encoder_layer = nn.TransformerEncoderLayer(d_model=d*4, nhead=1, batch_first=True, norm_first=True, device=device)
            # self.EEG_Transformer.append(nn.TransformerEncoder(encoder_layer, num_layers=1, enable_nested_tensor=False))
            # # self.EEG_Transformer.append(nn.Transformer(d_model=d*4, batch_first=True, norm_first=True))
            self.stages.append(SwinEncoder(d, 1, dev0, window_size=2))
            if i < 3:
                self.PatchMerging.append(nn.Conv2d(d, d*2, (4, 4), stride=(2, 2), padding=(1, 1), device=dev0))
                d *= 2
        # self.primaryCaps = PrimaryCaps(num_capsules=d, in_channels=8, out_channels=8, kernel_size=6, num_routes=8*1*124)
        # self.emotionCaps = EmotionCaps(num_capsules=nb_classes, num_routes=8*1*124, in_channels=d, out_channels=16)
        self.primaryCaps = PrimaryCap(d, 8, 48, 6, 1, 'same', 'v2', dev1)
        self.emotionCaps = EmotionCap(nb_classes, 16, 3, 8*48, 8, dev1)
    

    def forward(self, x):
        x = torch.permute(x, (0, 2, 1, 3))
        x = self.emd_expansion(x)
        x = torch.permute(x, (0, 2, 1, 3))
        x = self.PatchPartition(x)
        # bs, fs, hs, ws = x.shape
        for i in range(len(self.stages)):
            _, _, height, width = x.shape
            # x = x.view(bs, fs*(2**i)*4, -1)
            # x = x.permute(0, 2, 1)
            # x = self.PositionalEncoding[i](x) * math.sqrt(fs*(2**i)*4)
            # x = self.EEG_Transformer[i](x)
            x = self.stages[i](x)
            x = rearrange(x, 'b (h w) c -> b c h w', h=height, w=width)
            # x = x.permute(0, 2, 1)
            # x = x.reshape(bs, fs*(2**i), hs//(2**i), ws//(2**i))
            if i < len(self.PatchMerging):
                x = self.PatchMerging[i](x)
        if self.dev1 is not None:
            x = x.to(self.dev1)
        x = self.primaryCaps(x)
        x = self.emotionCaps(x)
        return torch.norm(x, dim=2).squeeze()



class MLFCapsNet(nn.Module):
    def __init__(self, nb_classes, Chans):
        super(MLFCapsNet, self).__init__()
        """ PyTorch Implementation of MLF-CapsNet """

        self.name = 'MLFCapsNet'
        
        self.conv = nn.Conv2d(1, 256, 6, 2, padding='valid') #TODO: test it
        self.relu = nn.ReLU()
        self.primaryCaps = PrimaryCap(256, 8, 32, 6, 1, 'same', 'v2')
        self.emotionCaps = EmotionCap(nb_classes, 16, 3, 32*1*128, 8)
    

    def forward(self, x):
        x = self.relu(self.conv(x))
        x = self.primaryCaps(x)
        x = self.emotionCaps(x)
        return torch.norm(x, dim=2).squeeze()



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


