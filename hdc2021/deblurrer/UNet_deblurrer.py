"""
Simple U-Net for deblurring

"""


import torch 
import torch.nn as nn 
import torch.nn.functional as F
import torchvision
import pytorch_lightning as pl
from skimage.transform import resize
from PIL import Image
import numpy as np
import pytesseract
from fuzzywuzzy import fuzz


#from dival.reconstructors.networks.unet import get_unet_model
from hdc2021.deblurrer.adapted_unet import UNet
from hdc2021.forward_model.blur import BlurOp 


def normalize(img):
    """
    Linear histogram normalization
    """
    arr = np.array(img, dtype=float)

    arr = (arr - arr.min()) * (255 / arr[:, :50].min())
    arr[arr > 255] = 255
    arr[arr < 0] = 0

    return Image.fromarray(arr.astype('uint8'), 'L')


def evaluateImage(img, trueText):

    # resize image to improve OCR
    img = img.detach().cpu().numpy()
    w, h = img.shape
    img = resize(img, (int(w / 2), int(h / 2)))

    img = Image.fromarray(np.uint8(img*255))
    img = normalize(img)
    # run OCR
    options = r'--oem 1 --psm 6 -c load_system_dawg=false -c load_freq_dawg=false  -c textord_old_xheight=0  -c textord_min_xheight=100 -c ' \
              r'preserve_interword_spaces=0'
    OCRtext = pytesseract.image_to_string(img, config=options)

    # removes form feed character  \f
    OCRtext = OCRtext.replace('\n\f', '').replace('\n\n', '\n')

    # split lines
    OCRtext = OCRtext.split('\n')

    # remove empty lines
    OCRtext = [x.strip() for x in OCRtext if x.strip()]

    # check if OCR extracted 3 lines of text
    #print('True text (middle line): %s' % trueText[1])

    if len(OCRtext) != 3:
        print('ERROR: OCR text does not have 3 lines of text!')
        #print(OCRtext)
        return 0.0
    else:
        #score = fuzz.ratio(trueText[1], OCRtext[1])
        if isinstance(trueText[1], list):
           score = fuzz.ratio(trueText[1][0], OCRtext[1])
        else:            
           score = fuzz.ratio(trueText[1], OCRtext[1])
        #print('OCR  text (middle line): %s' % OCRtext[1])
        #print('Score: %d' % score)

        return float(score)


class UNetDeblurrer(pl.LightningModule):
    def __init__(self, blurring_step, lr=1e-4, scales = 7, skip_channels=[4, 4, 8, 8, 16, 16, 32],
                channels=[8, 8, 16, 16, 32, 64, 128], kernel_size = (7, 7, 5, 5, 3, 3, 3),
                 use_sigmoid=True, batch_norm=True, init_bias_zero=True):
    #self, blurring_step=4, lr=1e-5, scales=6, skip_channels=4, channels=None,
    #             use_sigmoid=False, batch_norm=True, init_bias_zero=True):
        super().__init__()

        self.lr = lr
        
        #if channels is None:
        #    channels = (32, 32, 64, 64, 64, 64)

        self.init_bias_zero = init_bias_zero

        #save_hparams = {
        #    'lr': lr,
        #    'scales': scales,
        #    'skip_channels': skip_channels,
        #    'channels': channels,
        #    'use_sigmoid': use_sigmoid,
        #    'batch_norm': batch_norm,
        #    'init_bias_zero': init_bias_zero,
        #}
        save_hparams = {
            'lr': lr,
            'scales': scales,
            'skip_channels': skip_channels,
            'channels': channels,
            'kernel_size': kernel_size,
            'use_sigmoid': use_sigmoid,
            'batch_norm': batch_norm,
            'init_bias_zero': init_bias_zero,
        }
        self.save_hyperparameters(save_hparams)

        #self.net = get_unet_model(
        #    in_ch=1, out_ch=1, scales=scales, skip=skip_channels,
        #    channels=channels, use_sigmoid=use_sigmoid, use_norm=batch_norm)
        
        self.net = UNet(in_ch=1, out_ch=1, channels=channels[:scales], skip_channels=skip_channels[:scales], 
                kernel_size=kernel_size[:scales], 
                 use_sigmoid=use_sigmoid, use_norm=batch_norm)
                 
        if self.init_bias_zero:
            def weights_init(m):
                if isinstance(m, torch.nn.Conv2d):
                    m.bias.data.fill_(0.0)
            self.net.apply(weights_init)
            
        kernel_size = 701
        self.blur = BlurOp(inp_size=[1460, 2360],kernel_size=kernel_size)

        self.blur.load_state_dict(torch.load('/localdata/junickel/hdc2021/forward_model/weights/step_' + str(blurring_step) + ".pt"))
        self.blur.eval()
        for param in self.blur.parameters():
            param.requires_grad = False

        # normalize blurring kernel 
        self.blur.conv2d.weight.data = self.blur.conv2d.weight.data/torch.sum(self.blur.conv2d.weight.data)


    def forward(self, y):

        return self.net(y)

    def training_step(self, batch, batch_idx):
        # training_step defined the train loop.
        # It is independent of forward

        # if the batch is a dict, multiple dataloaders are used
        if isinstance(batch, dict):
            y, x, text = batch['hdc_loader']

            x_hat = self.forward(y) 
            
            weight_datasets_real = 0.9

            loss = weight_datasets_real*F.mse_loss(x_hat, x)

            y, x, text = batch['sim_loader']
            #with torch.no_grad():
            #    y = self.blur.forward(x)
            #    y = y  + 0.02*torch.randn(y.shape).to(y.device)

            x_hat = self.forward(y)

            loss = loss + (1.0-weight_datasets_real)*F.mse_loss(x_hat, x)
            self.log('train_loss', loss)
        else:
            y, x, text = batch
            x_hat = self.forward(y) 
            #if len(batch) == 3:
            #    y, x, text = batch
            #    x_hat = self.forward(y) 
               
            #if len(batch) == 2:
            #    x, text = batch
            #    with torch.no_grad():
            #        y = self.blur.forward(x)
            #        y = y  + 0.02*torch.randn(y.shape).to(y.device)

            #    x_hat = self.forward(y) 

            loss = F.mse_loss(x_hat, x)
            # Logging to TensorBoard by default
            self.log('train_loss', loss)
            
        return loss

    def validation_step(self, batch, batch_idx):
        # training_step defined the train loop.
        # It is independent of forward
        
        #if len(batch) == 3:
        #        y, x, text = batch
                
        #if len(batch) == 2:
        #    x, text = batch
        #    with torch.no_grad():
        #        y = self.blur.forward(x)
        #        y = y  + 0.02*torch.randn(y.shape).to(y.device)
        
        y, x, text = batch
        x_hat = self.net(y) 
        loss = F.mse_loss(x_hat, x)
        # Logging to TensorBoard by default
        self.log('val_loss', loss)
        self.last_batch = batch
        return loss 

    
    def validation_epoch_end(self, result):
        # no logging of histogram. Checkpoint gets too big
        #for name,params in self.named_parameters():
        #    self.logger.experiment.add_histogram(name, params, self.current_epoch)
        
        # if the batch is a dict, multiple dataloaders are used
        if isinstance(self.last_batch, dict):
            y, x, text = self.last_batch['hdc_loader']
        else:
            y, x, text = self.last_batch
            #if len(self.last_batch) == 3:
            #    y, x, text = self.last_batch
               
            #if len(self.last_batch) == 2:
            #    x, text = self.last_batch
            #    with torch.no_grad():
            #        y = self.blur.forward(x)
            #        y = y  + 0.02*torch.randn(y.shape).to(y.device)
        
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
        
        # calculate ocr score
        ocr_score = evaluateImage(x_hat[0][0], text)
        # Logging to TensorBoard by default
        self.log('ocr_score', ocr_score) 
         
          
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
