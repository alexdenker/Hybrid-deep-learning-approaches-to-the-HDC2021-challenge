"""
Learned Gradient Descent as a PyTorch Lightning module 
"""
import pytorch_lightning as pl

import torch 
import torch.nn as nn 
import torch.nn.functional as F

import torchvision

import pytesseract
from fuzzywuzzy import fuzz

import numpy as np 
from skimage.transform import resize
from PIL import Image

from hdc2021.forward_model.linear_blur import LinearBlurOp 
from hdc2021.deblurrer.adapted_unet import UNet

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
        if isinstance(trueText[1], list):
            score = fuzz.ratio(trueText[1][0], OCRtext[1])
        else:            
            score = fuzz.ratio(trueText[1], OCRtext[1])
        #print('OCR  text (middle line): %s' % OCRtext[1])
        #print('Score: %d' % score)

        return float(score)

class MultiScaleReconstructor(pl.LightningModule):
    def __init__(self, blurring_step, radius, 
                                      sigmas=[1e-2, 6e-3, 4e-3, 2e-3,5e-4], 
                                      step_sizes=[0.4, 0.3, 0.3, 0.3, 0.3],
                                      lr=1e-4, 
                                      batch_norm=True, 
                                      kernel_size = [3, 3, 5, 7, 9],
                                      skip_channels = [[16, 32, 32],
                                                       [16, 32, 32, 32], 
                                                       [16, 32, 32, 32], 
                                                       [4, 8, 8, 8], 
                                                       [4, 4, 4, 4]],
                                      channels =      [[32, 64, 64],
                                                       [32, 64, 64, 64],
                                                       [32, 64, 64, 64],
                                                       [8, 16, 16, 16],
                                                       [8, 8, 8, 8]],                         
                                      use_sigmoid=True, 
                                      init_x = True,
                                      n_memory = 2):
        super().__init__()

        
        self.blurring_step = blurring_step

        save_hparams = {
            'lr': lr,
            'batch_norm': batch_norm,
            'use_sigmoid': use_sigmoid,
            'sigmas': sigmas,
            'step_sizes': step_sizes,
            'channels': channels,
            'skip_channels': skip_channels,
            'radius': radius,
            'n_memory': n_memory,
            'init_x': init_x,
            'kernel_size': kernel_size
        }
        self.save_hyperparameters(save_hparams)

        self.lr = self.hparams.lr
        self.sigmas = self.hparams.sigmas
        self.step_sizes = self.hparams.step_sizes
        self.radius = self.hparams.radius
        self.n_memory = self.hparams.n_memory
        self.init_x = self.hparams.init_x 
        self.channels = self.hparams.channels
        self.skip_channels = self.hparams.skip_channels
        self.n_memory = self.hparams.n_memory
        self.kernel_size = self.hparams.kernel_size
        self.use_sigmoid = self.hparams.use_sigmoid
        self.batch_norm = self.hparams.batch_norm

        self.blur_1 = LinearBlurOp([1460, 2360], self.radius) # 4 : 0.025
        self.blur_2 = LinearBlurOp([730, 1180],  self.radius)
        self.blur_3 = LinearBlurOp([365, 590],   self.radius)
        self.blur_4 = LinearBlurOp([182, 295],   self.radius)
        self.blur_5 = LinearBlurOp([91, 147],    self.radius)


        self.nets = nn.ModuleList([UNet(in_ch=1 + self.n_memory, out_ch=1 + self.n_memory, channels=self.channels[i], skip_channels=self.skip_channels[i], 
                                                kernel_size=self.kernel_size[i], 
                                                use_sigmoid=self.use_sigmoid, use_norm=self.batch_norm)
                                        for i in range(5)])

    def forward(self, y, intermediate_outputs=False):

        y_list = [y] 
        for i in range(4):
            y_list.append(torch.nn.functional.interpolate(y_list[-1], scale_factor=1/2, mode='bilinear'))
        
        if intermediate_outputs:
            outs = []
        
        x_k = torch.zeros(y.shape[0], 1 + self.n_memory,
                                 *y_list[-1].shape[2:],
                                 device=y.device)
        if self.init_x:
            x_k[:] = self.blur_5.wiener(y_list[-1], sigma=self.sigmas[0])

        # Blur: 91 x 147
        x_grad = x_k[:, 0:1, ...] - self.step_sizes[0]*self.blur_5.wiener(self.blur_5.forward(x_k[:, 0:1, ...]) - y_list[4], sigma=self.sigmas[0])
        x_cat = torch.cat([x_grad, x_k[:,1:,...]], dim=1)
        x_k = self.nets[0](x_cat) 
        # Upsample 91 x 147 -> 182 x 295
        x_k = torch.nn.functional.interpolate(x_k, size=y_list[3].shape[2:], mode='nearest')
        if intermediate_outputs:
            outs.append(x_k[:, 0:1, ...])

        # Blur: 182 x 295
        x_grad = x_k[:, 0:1, ...] - self.step_sizes[1]*self.blur_4.wiener(self.blur_4.forward(x_k[:, 0:1, ...]) - y_list[3], sigma=self.sigmas[1])
        x_cat = torch.cat([x_grad, x_k[:,1:,...]], dim=1)
        x_k = self.nets[1](x_cat) 
        # Upsample 182 x 295 -> 365 x 590
        x_k = torch.nn.functional.interpolate(x_k, size=y_list[2].shape[2:], mode='nearest')
        if intermediate_outputs:
            outs.append(x_k[:, 0:1, ...])

        # Blur: 365 x 590
        x_grad = x_k[:, 0:1, ...] - self.step_sizes[2]*self.blur_3.wiener(self.blur_3.forward(x_k[:, 0:1, ...]) - y_list[2], sigma=self.sigmas[2])
        x_cat = torch.cat([x_grad, x_k[:,1:,...]], dim=1)
        x_k = self.nets[2](x_cat) 
        # Upsample 365 x 590 -> 730 x 1180
        x_k = torch.nn.functional.interpolate(x_k, size=y_list[1].shape[2:], mode='nearest')
        if intermediate_outputs:
            outs.append(x_k[:, 0:1, ...])

        # Blur: 730 x 1180
        x_grad = x_k[:, 0:1, ...] - self.step_sizes[3]*self.blur_2.wiener(self.blur_2.forward(x_k[:, 0:1, ...]) - y_list[1], sigma=self.sigmas[3])
        x_cat = torch.cat([x_grad, x_k[:,1:,...]], dim=1)
        x_k = self.nets[3](x_cat) 
        # Upsample 730 x 1180 -> 1460 x 2360
        x_k = torch.nn.functional.interpolate(x_k, size=y_list[0].shape[2:], mode='nearest')    
        if intermediate_outputs:
            outs.append(x_k[:, 0:1, ...])

        # Blur: 1460 x 2360
        x_grad = x_k[:, 0:1, ...] - self.step_sizes[4]*self.blur_1.wiener(self.blur_1.forward(x_k[:, 0:1, ...]) - y_list[0], sigma=self.sigmas[4])
        x_cat = torch.cat([x_grad, x_k[:,1:,...]], dim=1)
        x_k = self.nets[4](x_cat) 
        if intermediate_outputs:
            outs.append(x_k[:, 0:1, ...])

        if intermediate_outputs:
            return x_k[:, 0:1, ...], outs
        else:
            return x_k[:, 0:1, ...]

    def training_step(self, batch, batch_idx):
        # training_step defined the train loop.
        self.sim_batch = batch['sim_loader']


        if self.current_epoch < 150:
            y, x, text = batch['sim_loader']

            x_hat, outs = self.forward(y, intermediate_outputs=True) 
                
            x_list = [x] 
            for i in range(3):
                x_list.append(torch.nn.functional.interpolate(x_list[-1], scale_factor=1/2., mode='bilinear'))

            x_list.reverse()

            loss = F.mse_loss(x_hat, x)
            
            scale_loss = 0
            for j in range(len(x_list)):
                scale_loss += F.mse_loss(outs[j], x_list[j])

            loss += 0.1*scale_loss
            self.log('train_loss (sim)', loss)
        else:
            which_dataset = np.random.rand()
            if which_dataset < 0.25:
                y, x, text = batch['urban100_loader']

                x_hat, outs = self.forward(y, intermediate_outputs=True) 
                    
                x_list = [x] 
                for i in range(3):
                    x_list.append(torch.nn.functional.interpolate(x_list[-1], scale_factor=1/2., mode='bilinear'))

                x_list.reverse()

                loss = F.mse_loss(x_hat, x)
                
                scale_loss = 0
                for j in range(len(x_list)):
                    scale_loss += F.mse_loss(outs[j], x_list[j])

                loss += 0.3*scale_loss
                self.log('train_loss (urban)', loss)
            elif 0.25 <= which_dataset < 0.4:
                y, x, text = batch['sim_loader']

                x_hat, outs = self.forward(y, intermediate_outputs=True) 
                    
                x_list = [x] 
                for i in range(3):
                    x_list.append(torch.nn.functional.interpolate(x_list[-1], scale_factor=1/2., mode='bilinear'))

                x_list.reverse()

                loss = F.mse_loss(x_hat, x)
                
                scale_loss = 0
                for j in range(len(x_list)):
                    scale_loss += F.mse_loss(outs[j], x_list[j])

                loss += 0.3*scale_loss
                self.log('train_loss (sim)', loss)
            else:
                y, x, text = batch['hdc_loader']
                x_hat, outs = self.forward(y, intermediate_outputs=True) 
                    
                x_list = [x] 
                for i in range(3):
                    x_list.append(torch.nn.functional.interpolate(x_list[-1], scale_factor=1/2., mode='bilinear'))

                x_list.reverse()

                loss = F.mse_loss(x_hat, x)
                
                scale_loss = 0
                for j in range(len(x_list)):
                    scale_loss += F.mse_loss(outs[j], x_list[j])

                loss += 0.3*scale_loss

                self.log('train_loss (real)', loss)

        return loss

    def validation_step(self, batch, batch_idx):
        # training_step defined the train loop.
        y, x, text = batch
        
        x_hat = self.forward(y) 

        loss = F.mse_loss(x_hat, x)

        ocr_acc = evaluateImage(x_hat.cpu().numpy()[0][0], text)

        # Logging to TensorBoard by default
        self.log('val_loss', loss)
        self.log('val_ocr_acc', ocr_acc)

        self.last_batch = batch
        return ocr_acc 

    
    def validation_epoch_end(self, result):
        # no logging of histogram. Checkpoint gets too big
        #for name,params in self.named_parameters():
        #    self.logger.experiment.add_histogram(name, params, self.current_epoch)
        
        y, x, text = self.last_batch

               
        img_grid = torchvision.utils.make_grid(x, normalize=True,
                                               scale_each=True)

        self.logger.experiment.add_image(
            "ground truth", img_grid, global_step=self.current_epoch)
        
        blurred_grid = torchvision.utils.make_grid(y, normalize=True,
                                               scale_each=True)
        self.logger.experiment.add_image(
            "blurred image", blurred_grid, global_step=self.current_epoch)

        with torch.no_grad():
            x_hat, outs = self.forward(y, intermediate_outputs=True)

            for idx, xhat in enumerate(outs):

                reco_grid = torchvision.utils.make_grid(xhat, normalize=True,
                                                        scale_each=True)
                self.logger.experiment.add_image(
                    "deblurred scale {}".format(len(outs) - idx), reco_grid, global_step=self.current_epoch)


    def training_epoch_end(self, result):
        ysim, xsim, text = self.sim_batch

        img_grid_sim = torchvision.utils.make_grid(xsim, normalize=True,
                                               scale_each=True)

        self.logger.experiment.add_image(
            "sim_ground truth", img_grid_sim, global_step=self.current_epoch)
        
        blurred_grid_sim = torchvision.utils.make_grid(ysim, normalize=True,
                                               scale_each=True)
        self.logger.experiment.add_image(
            "sim_blurred image", blurred_grid_sim, global_step=self.current_epoch)
        
        with torch.no_grad():
            x_hat_sim = self.forward(ysim, intermediate_outputs=False)

            reco_grid_sim = torchvision.utils.make_grid(x_hat_sim, normalize=True,
                                                        scale_each=True)
            self.logger.experiment.add_image(
                    "sim_reconstruction", reco_grid_sim, global_step=self.current_epoch)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.nets.parameters(), lr=self.lr)
        return optimizer

