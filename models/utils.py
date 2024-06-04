import torch.nn as nn
import torch.nn.functional as F

class ConstrainedConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                    padding=0, dilation=1, groups=1, bias=True, nr=1.):
        super(ConstrainedConv2d, self).__init__(in_channels, out_channels, kernel_size, stride,
                                                padding, dilation, groups, bias)
        self.nr = nr

    def forward(self, input):
        if self.nr is None:
            return F.conv2d(input, self.weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)
        else:
            return F.conv2d(input, self.weight.clamp(min=-self.nr, max=self.nr), self.bias, self.stride,
                            self.padding, self.dilation, self.groups)
    

class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, bias=False):
        super(SeparableConv2d, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, 
                                groups=in_channels, bias=bias, padding=padding)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 
                                kernel_size=1, bias=bias)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out
    

class ConstrainedLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, norm_rate=.25):
        super(ConstrainedLinear, self).__init__(in_features, out_features, bias)
        self.norm_rate = norm_rate

    def forward(self, input):
        if self.norm_rate is None:
            return F.linear(input, self.weight, self.bias)
        else:
            return F.linear(input, self.weight.clamp(min=-self.norm_rate, max=self.norm_rate), self.bias)