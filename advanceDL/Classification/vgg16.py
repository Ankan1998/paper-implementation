import torch 
import torch.nn as nn

def double_conv(in_c,out_c):
    d_conv=nn.Sequential(
          nn.Conv2d(in_c,out_c,kernel_size=3,padding=1),
          nn.ReLU(inplace=True),
          nn.Conv2d(out_c,out_c,kernel_size=3,padding=1),
          nn.ReLU(inplace=True)
        )
    return d_conv


def triple_conv(in_c,out_c):
    t_conv=nn.Sequential(
          nn.Conv2d(in_c,out_c,kernel_size=3,padding=1),
          nn.ReLU(inplace=True),
          nn.Conv2d(out_c,out_c,kernel_size=3,padding=1),
          nn.ReLU(inplace=True),
          nn.Conv2d(out_c,out_c,kernel_size=3,padding=1),
          nn.ReLU(inplace=True)
        )
    return t_conv