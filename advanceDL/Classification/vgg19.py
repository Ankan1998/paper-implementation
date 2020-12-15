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


def quadruple_conv(in_c,out_c):
    t_conv=nn.Sequential(
          nn.Conv2d(in_c,out_c,kernel_size=3,padding=1),
          nn.ReLU(inplace=True),
          nn.Conv2d(out_c,out_c,kernel_size=3,padding=1),
          nn.ReLU(inplace=True),
          nn.Conv2d(out_c,out_c,kernel_size=3,padding=1),
          nn.ReLU(inplace=True),
          nn.Conv2d(out_c,out_c,kernel_size=3,padding=1),
          nn.ReLU(inplace=True)
        )
    return t_conv




class vgg19(nn.Module):

  def __init__(self,num_of_classes):
    super().__init__()

    self.num_of_classes=num_of_classes

    # Conv Layer
    self.mpool2d=nn.MaxPool2d(kernel_size=2, stride=2)
    self.dconv1=double_conv(3,64)
    self.dconv2=double_conv(64,128)
    self.qconv1=quadruple_conv(128,256)
    self.qconv2=quadruple_conv(256,512)
    self.qconv3=quadruple_conv(512,512)

    #Linear Layer
    self.flat=nn.Flatten()
    self.relu=nn.ReLU()
    self.fc1=nn.Linear(7*7*512,4096)
    self.fc2=nn.Linear(4096,4096)
    self.last=nn.Linear(4096,self.num_of_classes)




  def forward(self,x):
    #print("start",x.shape)
    x=self.dconv1(x)
    x=self.mpool2d(x)
    x=self.dconv2(x)
    x=self.mpool2d(x)
    x=self.qconv1(x)
    x=self.mpool2d(x)
    x=self.qconv2(x)
    x=self.mpool2d(x)
    x=self.qconv3(x)
    x=self.mpool2d(x)
    x=self.flat(x)
    x=self.fc1(x)
    x=self.relu(x)
    x=self.fc2(x)
    x=self.relu(x)
    x=self.last(x)

    #print("final",x.shape)

    return x