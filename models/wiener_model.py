import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F
from models.util import show_output_tensor

class MultiWienerDeconvolution3D(nn.Module):
    """
    Performs Wiener Deconvolution in the frequency domain for each psf.
    
    Input: initial_psfs of shape (Y, X, C), initial_K has shape (1, 1, C) for each psf.
    """
    
    def __init__(self, initial_psfs, initial_Ks):
        super(MultiWienerDeconvolution3D, self).__init__()
        initial_psfs = torch.tensor(initial_psfs, dtype=torch.float32)
        initial_Ks = torch.tensor(initial_Ks, dtype=torch.float32)

        self.psfs = nn.Parameter(initial_psfs, requires_grad =True)
        self.Ks = nn.Parameter(initial_Ks, requires_grad =True) #NEEED RELU CONSTRAINT HERE K is constrained to be nonnegative
        
    def forward(self, y):
        # Y preprocessing, Y is shape (N, C,H, W)
        h, w = y.shape[-3:-1]
        y = y.type(torch.complex64)

    
        # Pad Y
        padding = ((0, 0), 
                   (int(np.ceil(h / 2)), int(np.floor(h / 2))),
                   (int(np.ceil(w / 2)), int(np.floor(w / 2))),
                    (0, 0))

        # Temporarily transpose y since we cannot specify axes for fft2d
        Y=torch.fft.fft2(y)

        # Components preprocessing, psfs is shape (C,H, W)
        psf = self.psfs.type(torch.complex64)
        h_psf, w_psf = self.psfs.shape[0:2]

        # Pad psf
        padding_psf = (
                   (int(np.ceil(h_psf / 2)), int(np.floor(h_psf / 2))),
                   (int(np.ceil(w_psf / 2)), int(np.floor(w_psf / 2))),
                    (0, 0))

        H_sum = torch.fft.fft2(self.psfs)

        X=(torch.conj(H_sum)*Y)/ (torch.square(torch.abs(H_sum))+100*self.Ks)#, dtype=tf.complex64)
    
        x=torch.real((torch.fft.ifftshift(torch.fft.ifft2(X), dim=(-2, -1))))
        

        return x
    
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'initial_psfs': self.psfs.numpy(),
            'initial_Ks': self.Ks.numpy()
        })
        return config
    
    
class WienerDeconvolution3D(nn.Module):
    """
    Performs Wiener Deconvolution in the frequency domain for each psf.
    
    Input: initial_psfs of shape (Y, X, C), initial_K has shape (1, 1, C) for each psf.
    """
    
    def __init__(self, initial_psfs, initial_Ks):
        super(WienerDeconvolution3D, self).__init__()
        initial_psfs = torch.tensor(initial_psfs, dtype=torch.float32).cuda()
        initial_Ks = torch.tensor(initial_Ks, dtype=torch.float32).cuda()

        self.psfs = initial_Ks#nn.Parameter(initial_psfs, requires_grad =True)
        self.Ks = initial_Ks#nn.Parameter(initial_Ks, requires_grad =True) #NEEED RELU CONSTRAINT HERE K is constrained to be nonnegative
        
    def weiner_filter(self,img,kernel,K):
        #kernel /= torch.sum(kernel)
        #dummy = torch.clone(img)
        dummy = torch.fft.fft2(img).cuda()
        #kernel = torch.unsqueeze(kernel,0).repeat(8,1,1,1).cuda()
        #print(kernel.shape)
        kernel = torch.fft.fft2(kernel).cuda()
        #print(kernel.shape)
        kernel = torch.conj(kernel).cuda()/(torch.abs(kernel.cuda())**2 + K.cuda())
        dummy = dummy * kernel
        dummy = torch.fft.ifftshift(torch.abs(torch.fft.ifft2(dummy)),dim=(-2, -1))
        return dummy    

    def forward(self, y):
        return self.weiner_filter(y,self.psfs,self.Ks)
    '''
        # Y preprocessing, Y is shape (N, C,H, W)
        h, w = y.shape[-2:]
        y = y.type(torch.complex64)

    
        # Pad Y
        padding = ((0, 0), 
                   (int(np.ceil(h / 2)), int(np.floor(h / 2))),
                   (int(np.ceil(w / 2)), int(np.floor(w / 2))),
                    (0, 0))

        # Temporarily transpose y since we cannot specify axes for fft2d
        Y=torch.fft.fft2(y)

        # Components preprocessing, psfs is shape (C,H, W)
        
        h_psf, w_psf = self.psfs.shape[-2:]

        # Pad psf
        padding_psf = (
                   (int(np.ceil(h_psf / 2)), int(np.floor(h_psf / 2))),
                   (int(np.ceil(w_psf / 2)), int(np.floor(w_psf / 2))),
                    (0, 0))

        #padding_psf_test = (int(np.ceil(h / 2)), int(np.floor(h / 2)), int(np.ceil(w / 2)), int(np.floor(w / 2)))
        #padding_psf_test = (1,1,1,1)
        #self.psfs = torch.nn.functional.pad(, padding_psf_test, mode='constant',value=0)
        psf = self.psfs.type(torch.complex64)

        H_sum = torch.fft.fft2(self.psfs)
        #print(psf[0,120:130,120:130])

        #print(H_sum.shape, Y.shape, self.Ks.shape)
        #show_output_tensor(Y)
        #show_output_tensor(H_sum)
        X=(torch.conj(H_sum)*Y)/ (torch.square(torch.abs(H_sum)*Y)+1000*self.Ks)#, dtype=tf.complex64)
    
        x=torch.real((torch.fft.ifftshift(torch.fft.ifft2(X), dim=(-2, -1))))
        output = ((x-torch.min(x))/(torch.max(x)-torch.min(x)))
        #print(output.shape)
        #show_output_tensor(output)

        return output
    '''
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'initial_psfs': self.psfs.numpy(),
            'initial_Ks': self.Ks.numpy()
        })
        return config
    
    
class MyEnsemble2d(nn.Module):
    def __init__(self, wiener_model, unet_model):
        super(MyEnsemble2d, self).__init__()
        self.wiener_model = wiener_model
        self.unet_model = unet_model
    def forward(self, x):
        wiener_output = self.wiener_model(x)
        wiener_output = wiener_output/torch.max(wiener_output)
        final_output = self.unet_model(wiener_output)
        return final_output

class MyEnsemble(nn.Module):
    def __init__(self, wiener_model, unet_model):
        super(MyEnsemble, self).__init__()
        self.wiener_model = wiener_model
        self.unet_model = unet_model
    def forward(self, x):
        wiener_output = self.wiener_model(x)
        final_output = self.unet_model(wiener_output)
        return final_output