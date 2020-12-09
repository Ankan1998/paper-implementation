import torch
import torch.nn as nn



# This function is for double convolution function
def double_down_conv(in_c,out_c):
    d_conv_seq=nn.Sequential(
          nn.Conv2d(in_c,out_c,kernel_size=3),
          nn.ReLU(inplace=True),
          nn.Conv2d(out_c,out_c,kernel_size=3),
          nn.ReLU(inplace=True)
        )
    return d_conv_seq


# Cropping tensor of image from encoder side to concat to decoder side
def crop_tensor(target,original):

  target_size=target.size()[2]
  original_size=original.size()[2]
  diff=original_size-target_size
  diff=diff//2
  return original[:,:,diff:original_size-diff,diff:original_size-diff]


class UNet(nn.Module):


  def __init__(self):
    super().__init__()
    # Layers 
    # 1st part
    self.mpool2d=nn.MaxPool2d(kernel_size=2, stride=2)
    self.dconv1=double_down_conv(1,64)
    self.dconv2=double_down_conv(64,128)
    self.dconv3=double_down_conv(128,256)
    self.dconv4=double_down_conv(256,512)
    self.dconv5=double_down_conv(512,1024)


    # 2nd part
    self.tconv1=nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=2, stride=2)
    self.tconv2=nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=2, stride=2)
    self.tconv3=nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=2, stride=2)
    self.tconv4=nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=2, stride=2)

    self.double_u_conv1=double_down_conv(1024,512)
    self.double_u_conv2=double_down_conv(512,256)
    self.double_u_conv3=double_down_conv(256,128)
    self.double_u_conv4=double_down_conv(128,64)


    self.out=nn.Conv2d(64,2,kernel_size=1)
    

  def forward(self,image):

    # Encoder part
    x1=self.dconv1(image) #
    x2=self.mpool2d(x1)
    x3=self.dconv2(x2) #
    x4=self.mpool2d(x3)
    x5=self.dconv3(x4) #
    x6=self.mpool2d(x5)
    x8=self.dconv4(x6) #
    x9=self.mpool2d(x8)
    x10=self.dconv5(x9)

    # Decoder Part

    x11=self.tconv1(x10)
    xnew1=crop_tensor(x11,x8)
    x12=self.double_u_conv1(torch.cat([x11,xnew1],1))


    x13=self.tconv2(x12)
    xnew2=crop_tensor(x13,x5)
    x14=self.double_u_conv2(torch.cat([x13,xnew2],1))


    x15=self.tconv3(x14)
    xnew3=crop_tensor(x15,x3)
    x16=self.double_u_conv3(torch.cat([x15,xnew3],1))


    x17=self.tconv4(x16)
    xnew4=crop_tensor(x17,x1)
    x18=self.double_u_conv4(torch.cat([x17,xnew4],1))

    print("x18",x18.size())
    xfinal=self.out(x18)
    print("xfinal",xfinal.size())

    return xfinal
    
