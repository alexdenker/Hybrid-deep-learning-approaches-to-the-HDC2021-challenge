
import os
from pathlib import Path
#os.environ["CUDA_VISIBLE_DEVICES"] = "4"

import pytorch_lightning as pl
from PIL import Image

import torch
import torch.nn.functional as F
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning import loggers as pl_loggers
from dival.measure import PSNRMeasure, SSIMMeasure
import numpy as np
from torchvision.datasets import STL10
from torchvision.transforms import ToTensor, Grayscale, Compose

#from hdc2021_challenge.utils.blurred_dataset import BlurredDataModule
#from hdc2021_challenge.deblurrer.UNet_deblurrer import UNetDeblurrer

from hdc2021.utils.blurred_dataset import HDCDataset, HDCDatasetTest
from hdc2021.deblurrer.DIP_deblurrer import DIPDeblurrer
from torch.utils.data import DataLoader, Dataset
from hdc2021.deblurrer.UNet_deblurrer import UNetDeblurrer
from hdc2021.forward_model.blur import BlurOp 
from hdc2021.deblurrer.DIP_deblurrer import evaluateImage

class STL10Dataset(Dataset):
    def __init__(self,blur, idx):
        
        self.stl10 = STL10('/localdata/junickel/hdc2021/stl10', transform=Compose([Grayscale(),
                                                                        ToTensor()]), 
                           download = False)
        self.stl10_data = self.stl10[idx][0].unsqueeze(0)
        self.stl10_data = torch.nn.functional.interpolate(self.stl10_data, size = [1460,2360])
        
    def __len__(self):
        
        return 1

    def __getitem__(self, idx):
        
        datapoint = (blur(self.stl10_data)[0], self.stl10_data[0])

        return datapoint



# Number of DIP epochs per Step
EPOCH_DICT = {
    4 : 2000,
    9 : 4000,
    14 : 5000,
    19 : 5000
}

# Tolerance above which the DIP post-processing is started. The values gets
# higher for every few epochs, since our forward model is also less accurate
# for higher steps
DATA_TOLERANCE = {
    4 : 0.007,
    9 : 0.015,
    14 : 0.015,
    19 : 0.035 
}

blurring_step = 14

PSNR = PSNRMeasure(data_range = 1.)
SSIM = SSIMMeasure(data_range = 1.)

ocr_mean = []
ocri_mean = []
discrepancy_mean = []
psnr_mean = []
ssim_mean = []

# Load forward model
blur = BlurOp(inp_size=[1460, 2360],kernel_size=701)
blur.load_state_dict(torch.load('/localdata/junickel/hdc2021/forward_model/weights/step_' + str(blurring_step) + ".pt"))
blur.eval()
blur.requires_grad_(False)
# normalize blurring kernel 
blur.conv2d.weight.data = blur.conv2d.weight.data/torch.sum(blur.conv2d.weight.data)

# Load trained U-Net 
u_net_module = UNetDeblurrer.load_from_checkpoint(Path('/localdata/junickel/hdc2021/baseline_unet/weights/step_' + str(blurring_step) + '.ckpt'), blurring_step = blurring_step) 
u_net_module.eval()
u_net = u_net_module.net

#dataset = BlurredDataModule(batch_size=1, blurring_step=blurring_step)

font = 'Sanity' # 'Times', 'Sanity'
dataset = HDCDatasetTest(step=blurring_step, subset='sanity', font=font, shift_bg=None)

for num_datapoint in range(15,16): #len(dataset)):
    
    datapoint = torch.utils.data.Subset(dataset, [num_datapoint])
    
    # STL10 dataset
    #datapoint = STL10Dataset(blur=blur,idx=num_datapoint)
       
    # Check data discrepancy
    with torch.no_grad():
        x_hat = u_net_module(datapoint[0][0].unsqueeze(0))
        y_hat = blur(x_hat)
        discrepancy = F.mse_loss(y_hat, datapoint[0][0].unsqueeze(0))
        print(discrepancy)
       
    if discrepancy < DATA_TOLERANCE[blurring_step]:
        print('Initial output within tolerance. No postprocessing needed.')
        reconstruction = torch.clip(x_hat, 0, 1)
    else: 
        # Train DIP
        print('Start training of DIP')
        checkpoint_callback = ModelCheckpoint(
        dirpath=None,
        save_top_k=1,
        verbose=True,
        monitor='train_loss_deblurred',
        mode='max',
        #prefix=''
        )
        
        base_path = '/localdata/junickel/hdc2021'
        experiment_name = 'dip_deblurring'
        #blurring_step = "step_" + str(blurring_step)
        path_parts = [base_path, experiment_name, "step_" + str(blurring_step)]
        log_dir = os.path.join(*path_parts)
        tb_logger = pl_loggers.TensorBoardLogger(log_dir)
        
        trainer_args = {'accelerator': 'ddp',
                        'gpus': 1,
                        'default_root_dir': log_dir,
                        'callbacks': [checkpoint_callback],
                        'benchmark': False,
                        'fast_dev_run': False,
                        'gradient_clip_val': 1.0,
                        'logger': tb_logger,
                        'log_every_n_steps': 10,
                        'auto_scale_batch_size': 'binsearch'}#,
                        #'accumulate_grad_batches': 6}#,}
                        # 'log_gpu_memory': 'all'} # might slow down performance (unnecessary uses only the output of nvidia-smi)
    
        reconstructor = DIPDeblurrer(blurring_step=blurring_step, lr=1e-4,gamma=1e-11,use_sigmoid=True)
        
        trainer = pl.Trainer(max_epochs=EPOCH_DICT[blurring_step], **trainer_args)
        
        trainer.fit(reconstructor, DataLoader(datapoint, batch_size=1))
        
        reconstruction = torch.clip(reconstructor(datapoint[0][0].unsqueeze(0)),0,1)
        
    # Evaluation
    
    # OCR score
    if len(datapoint[0][2]) > 0:
      ocr_score, ocri_score = evaluateImage(reconstruction[0][0],datapoint[0][1].squeeze(),datapoint[0][2])
      ocr_mean.append(ocr_score)
      ocri_mean.append(ocri_score)
    else:
      ocr_score = 0
      ocr_mean.append(0)
      ocri_score = 0
      ocri_mean.append(0)
    
    # Data discrepancy
    y_hat = blur(reconstruction)
    discrepancy = F.mse_loss(y_hat, datapoint[0][0].unsqueeze(0))
    discrepancy_mean.append(discrepancy.detach().numpy())
    
    # psnr
    psnr_reco = PSNR(reconstruction.detach().numpy().squeeze(),
                     datapoint[0][1].detach().numpy().squeeze())
    psnr_mean.append(psnr_reco)
    ssim_reco = SSIM(reconstruction.detach().numpy().squeeze(),
                     datapoint[0][1].detach().numpy().squeeze())
    ssim_mean.append(ssim_reco)
    
    # Save data
    os.makedirs('/localdata/junickel/hdc2021/results_dip/step_' + str(blurring_step), exist_ok=True)
    
    im = Image.fromarray(datapoint[0][0].detach().cpu().numpy()[0]*255.).convert("L")
    im.save('/localdata/junickel/hdc2021/results_dip/step_' + str(blurring_step) + '/blurred_' + str(num_datapoint) + '_font_' + str(font) + '.PNG')
    
    im = Image.fromarray(datapoint[0][1].detach().cpu().numpy()[0]*255.).convert("L")
    im.save('/localdata/junickel/hdc2021/results_dip/step_' + str(blurring_step) + '/ground_truth_' + str(num_datapoint) + '_font_' + str(font) + '.PNG')
    
    im = Image.fromarray(reconstruction.detach().cpu().numpy()[0][0]*255.).convert("L")
    im.save('/localdata/junickel/hdc2021/results_dip/step_' + str(blurring_step) + '/reconstruction_' + str(num_datapoint) + '_font_' + str(font) + '.PNG')
    
    with open('/localdata/junickel/hdc2021/results_dip/step_' + str(blurring_step) + '/metrics_' + str(num_datapoint) + '_font_' + str(font) + '.txt', 'w') as f:
        f.write('ocr_score: ' + str(ocr_score) + '\n' + \
                'ocri_score: ' + str(ocri_score) + '\n' + \
                'discrepancy: ' + str(discrepancy) + '\n' +\
                'psnr: ' + str(psnr_reco) + '\n' +\
                'ssim: ' + str(ssim_reco))
 
mean_ocr = np.array(ocr_mean).mean()  
std_ocr = np.array(ocr_mean).std()
mean_ocri = np.array(ocri_mean).mean()    
std_ocri = np.array(ocri_mean).std() 
discrepancy_mean = np.array(discrepancy_mean).mean() 
mean_psnr = np.mean(psnr_mean)
std_psnr = np.std(psnr_mean)
mean_ssim = np.mean(ssim_mean)
std_ssim = np.std(ssim_mean)   

with open('/localdata/junickel/hdc2021/results_dip/step_' + str(blurring_step) + '/metrics_overall_font_' + str(font) + '.txt', 'w') as f:
    f.write('ocr_score: ' + str(mean_ocr) + ' +- ' + str(std_ocr) + '\n' + \
            'ocri_score: ' + str(mean_ocri) + ' +- ' + str(std_ocri) + '\n' + \
            'discrepancy: ' + str(discrepancy_mean) + '\n' + \
            'psnr: ' + str(mean_psnr) + ' +- ' + str(std_psnr) + '\n' + \
            'ssim: ' + str(mean_ssim) + ' +- ' + str(std_ssim))
 

print('ocr_score: ' + str(mean_ocr) + ' +- ' + str(std_ocr) + '\n' + \
      'ocri_score: ' + str(mean_ocri) + ' +- ' + str(std_ocri) + '\n' + \
      'discrepancy: ' + str(discrepancy_mean) + '\n' + \
      'psnr: ' + str(mean_psnr) + ' +- ' + str(std_psnr) + '\n' + \
      'ssim: ' + str(mean_ssim) + ' +- ' + str(std_ssim))