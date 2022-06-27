"""
DIP for deblurring

"""
from pathlib import Path

import torch 
import torch.nn.functional as F

import torchvision
import pytorch_lightning as pl
from skimage.transform import resize
from PIL import Image
import numpy as np
import pytesseract
from fuzzywuzzy import fuzz

#from dival.reconstructors.networks.unet import get_unet_model
from hdc2021.forward_model.blur import BlurOp 
from hdc2021.deblurrer.UNet_deblurrer import UNetDeblurrer
from hdc2021.deblurrer.adapted_unet import UNet
from hdc2021.utils.blurred_dataset import _psf_shift

def normalize(img):
    """
    Linear histogram normalization
    """
    arr = np.array(img, dtype=float)

    arr = (arr - arr.min()) * (255 / arr[:, :50].min())
    arr[arr > 255] = 255
    arr[arr < 0] = 0

    return Image.fromarray(arr.astype('uint8'), 'L')


def evaluateImage(img, gt, trueText):

    # resize image to improve OCR
    img = img.detach().cpu().numpy()
    gt = gt.detach().cpu().numpy()
    
    w, h = img.shape
    img = resize(img, (int(w / 2), int(h / 2)))
    gt = resize(gt, (int(w / 2), int(h / 2)))

    img = Image.fromarray(np.uint8(img*255))
    img = normalize(img)

    gt = Image.fromarray(np.uint8(gt*255))
    gt = normalize(gt)
    
    # run OCR
    options = r'--oem 1 --psm 6 -c load_system_dawg=false -c load_freq_dawg=false  -c textord_old_xheight=0  -c textord_min_xheight=100 -c ' \
              r'preserve_interword_spaces=0'
    OCRtext = pytesseract.image_to_string(img, config=options)
    OCRGTtext = pytesseract.image_to_string(gt, config=options)

    # removes form feed character  \f
    OCRtext = OCRtext.replace('\n\f', '').replace('\n\n', '\n')
    OCRGTtext = OCRGTtext.replace('\n\f', '').replace('\n\n', '\n')

    # split lines
    OCRtext = OCRtext.split('\n')
    OCRGTtext = OCRGTtext.split('\n')

    # remove empty lines
    OCRtext = [x.strip() for x in OCRtext if x.strip()]
    OCRGTtext = [x.strip() for x in OCRGTtext if x.strip()]

    # check if OCR extracted 3 lines of text
    #print('True text (middle line): %s' % trueText[1])
    #print(trueText, OCRtext, OCRGTtext)
    
    if len(OCRtext) != 3 and len(OCRGTtext) == 3:
        print('ERROR: OCR text does not have 3 lines of text!')
        return (0.0, 0.0)
    else:
        if isinstance(trueText[1], list):
           score = fuzz.ratio(trueText[1][0], OCRtext[1])
        else:            
           score = fuzz.ratio(trueText[1], OCRtext[1])

        score_gt = fuzz.ratio(OCRGTtext[1], OCRtext[1])

        return (float(score), float(score_gt))


def tv_loss(x):
    """
    Anisotropic TV loss.
    """
    dh = torch.abs(x[..., :, 1:] - x[..., :, :-1])
    dw = torch.abs(x[..., 1:, :] - x[..., :-1, :])
    return torch.sum(dh[..., :-1, :] + dw[..., :, :-1])

def tv_iso(x):
    epsilon = 1e-10
    y1 = x[...,1:,:] - x[...,:-1,:]
    y2 = x[...,:,1:] - x[...,:,:-1]
    Y = torch.zeros(x.shape[-2:],dtype=x.dtype,device=x.device)
    Y[:-1,:-1] = y1[...,:,:-1]**2
    Y[:-1,:-1] = Y[:-1,:-1] + y2[...,:-1,:]**2
    Y[:-1,:-1] = torch.sqrt(Y[:-1,:-1] + epsilon)
    Y[:-1,-1] = torch.abs(y1[...,:,-1])
    Y[-1,:-1] = torch.abs(y2[...,-1,:])
    return Y.sum()

def perona_malik(x, T=0.15):
    epsilon = 1e-10
    y1 = x[...,1:,:] - x[...,:-1,:]
    y2 = x[...,:,1:] - x[...,:,:-1]
    Y = torch.zeros(x.shape[-2:],dtype=x.dtype,device=x.device)
    Y[:-1,:-1] = y1[...,:,:-1]**2
    Y[:-1,:-1] = Y[:-1,:-1] + y2[...,:-1,:]**2
    Y[:-1,:-1] = torch.sqrt(Y[:-1,:-1] + epsilon)
    Y[:-1,-1] = torch.abs(y1[...,:,-1])
    Y[-1,:-1] = torch.abs(y2[...,-1,:])
    
    Y = 1/2*T**2*(1 - torch.exp(-1/T**2*Y**2))
    return Y.sum()
    
def KL_divergence(ypred, ytrue):
    epsilon = 1e-10
    kl_div_pixel = ytrue * (torch.log(ytrue + epsilon) - torch.log(ypred + epsilon)) + ypred - ytrue
    return kl_div_pixel.sum()

class DIPDeblurrer(pl.LightningModule):
    def __init__(self, blurring_step, lr=1e-4, scales = 7, 
                 skip_channels=[4, 4, 8, 8, 16, 16, 32],
                 channels=[8, 8, 16, 16, 32, 64, 128], kernel_size = (7, 7, 5, 5, 3, 3, 3),
                 use_sigmoid=True, batch_norm=True, init_bias_zero=True, gamma=0.0, kappa=1e-6):
                 
        super().__init__()

        self.lr = lr
        
        if channels is None:
            channels = (32, 32, 64, 64, 64, 64)

        self.init_bias_zero = init_bias_zero

        save_hparams = {
            'lr': lr,
            'scales': scales,
            'skip_channels': skip_channels,
            'channels': channels,
            'use_sigmoid': use_sigmoid,
            'batch_norm': batch_norm,
            'init_bias_zero': init_bias_zero,
        }
        self.save_hyperparameters(save_hparams)

        #self.net = get_unet_model(
        #    in_ch=1, out_ch=1, scales=scales, skip=skip_channels,
        #    channels=channels, use_sigmoid=use_sigmoid, use_norm=batch_norm)

        #if self.init_bias_zero:
        #    def weights_init(m):
        #        if isinstance(m, torch.nn.Conv2d):
        #            m.bias.data.fill_(0.0)
        #    self.net.apply(weights_init)
        
        # Forward blur operator
        blur = BlurOp(inp_size=[1460, 2360],kernel_size=701)
        blur.load_state_dict(torch.load('/localdata/junickel/hdc2021/forward_model/weights/step_' + str(blurring_step) + ".pt"))
        blur.eval()
        blur.requires_grad_(False)
        # normalize blurring kernel 
        blur.conv2d.weight.data = blur.conv2d.weight.data/torch.sum(blur.conv2d.weight.data)
        self.blur = blur
        #self.blur = blur_op
        
        # Load trained U-Net 
        u_net_module = UNetDeblurrer.load_from_checkpoint(Path('/localdata/junickel/hdc2021/baseline_unet/weights/step_' + str(blurring_step) + '.ckpt'), blurring_step = blurring_step) 
        self.net = u_net_module.net
        
        self.gamma = gamma
        self.kappa = kappa
        
        # Load psf_shift images
        y_back, x_back = _psf_shift(blurring_step)
        self.y_back = y_back.unsqueeze(0).unsqueeze(0).to('cuda')
        self.x_back = x_back.unsqueeze(0).unsqueeze(0).to('cuda')


    def forward(self, y):

        return self.net(y)

    def training_step(self, batch, batch_idx):
        # training_step defined the train loop.
        # It is independent of forward

        if len(batch) == 3:
        
            y, x, text = batch
            x_hat = self.forward(y) 
            
            y_hat = self.blur.forward(x_hat - self.x_back)
            
            #loss = KL_divergence(y_hat, y) + self.gamma*perona_malik(x_hat) 
            #F.mse_loss(self.blur.conv2d(x_hat), self.blur.undistort(y))
            #F.kl_div(torch.log(y_hat), y, reduction = 'batchmean') + self.gamma*tv_iso(x_hat) 
            #loss = F.mse_loss(y_hat, y - self.y_back) + self.gamma*tv_iso(x_hat)
            l1_loss = torch.nn.L1Loss()
            loss = 0.5*l1_loss(y_hat, y - self.y_back) + 0.5*F.mse_loss(y_hat, y - self.y_back) + self.kappa*tv_loss(x_hat)
            
            loss_deblurred = F.mse_loss(x_hat, x)
            
            # calculate ocr score
            if text != ('',):
                ocr_score, ocri_score = evaluateImage(x_hat[0][0], x[0][0], text)
                self.log('ocr_score', ocr_score)
                    
            # Logging to TensorBoard by default
            self.log('train_loss', loss)
            self.log('train_loss_deblurred', loss_deblurred)
            self.log('mse', F.mse_loss(y_hat, y - self.y_back))
            self.log('tv', tv_iso(x_hat))
            
        elif len(batch) == 2:
            y, x = batch
            x_hat = self.forward(y) 
            
            y_hat = self.blur.forward(x_hat - self.x_back)
            
            #loss = KL_divergence(y_hat, y) + self.gamma*perona_malik(x_hat) 
            #F.mse_loss(self.blur.conv2d(x_hat), self.blur.undistort(y))
            #F.kl_div(torch.log(y_hat), y, reduction = 'batchmean') + self.gamma*tv_iso(x_hat) 
            #loss = F.mse_loss(y_hat, y - self.y_back) + self.gamma*tv_iso(x_hat)
            l1_loss = torch.nn.L1Loss()
            loss = 0.5*l1_loss(y_hat, y - self.y_back) + 0.5*F.mse_loss(y_hat, y - self.y_back) + self.kappa*tv_loss(x_hat)
            
            loss_deblurred = F.mse_loss(x_hat, x)
            
            # Logging to TensorBoard by default
            self.log('train_loss', loss)
            self.log('train_loss_deblurred', loss_deblurred)
            self.log('mse', F.mse_loss(y_hat, y - self.y_back))
            self.log('tv', tv_iso(x_hat))
            
       
        self.last_batch = batch

        return loss

    
    def training_epoch_end(self, result):
        # no logging of histogram. Checkpoint gets too big
        #for name,params in self.named_parameters():
        #    self.logger.experiment.add_histogram(name, params, self.current_epoch)
        
        # if the batch is a dict, multiple dataloaders are used
        if len(self.last_batch) == 3:
            y, x, text = self.last_batch
        elif len(self.last_batch) == 2:
            y, x = self.last_batch
        
        img_grid = torchvision.utils.make_grid(x, normalize=True,
                                               scale_each=True)

        self.logger.experiment.add_image(
            "ground truth", img_grid, global_step=self.current_epoch)
        
        blurred_grid = torchvision.utils.make_grid(y, normalize=True,
                                               scale_each=True)
        self.logger.experiment.add_image(
            "blurred image", blurred_grid, global_step=self.current_epoch)

        with torch.no_grad():
            x_hat = self.forward(y)

            reco_grid = torchvision.utils.make_grid(x_hat, normalize=True,
                                                    scale_each=True)
            self.logger.experiment.add_image(
                "deblurred", reco_grid, global_step=self.current_epoch)
          
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
