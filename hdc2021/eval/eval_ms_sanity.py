"""
Eval PSNR and SSIM of LGD on Sanity images

"""

import os
from pathlib import Path

import torch
import yaml
from dival.util.plot import plot_images

import matplotlib.pyplot as plt 

from dival.measure import PSNRMeasure, SSIMMeasure

PSNR = PSNRMeasure(data_range=1.)
SSIM = SSIMMeasure(data_range=1.)

from tqdm import tqdm
import numpy as np 

import pytesseract
from fuzzywuzzy import fuzz

from skimage.transform import resize
from PIL import Image

from hdc2021.utils.blurred_dataset import BlurredDataModule, MultipleBlurredDataModule, HDCDatasetTest
from hdc2021.deblurrer.MultiScale_GDv4 import MultiScaleReconstructor



blurring_step = 19

step_to_radius = {4 : 0.015,
                  9 : 0.03,
                  14: 0.043,  #0.04
                  19: 0.0875}


save_report = True 

base_path = "/localdata/AlexanderDenker/deblurring_experiments"
experiment_name = 'step_' + str(blurring_step)  
version = 'version_3'

path_parts = [base_path, 'multi_scale', experiment_name, 'default',
            version, 'checkpoints']
chkp_path = os.path.join(*path_parts)
chkp_path = os.path.join(chkp_path, os.listdir(chkp_path)[0])
print(chkp_path)

reconstructor = MultiScaleReconstructor.load_from_checkpoint(chkp_path, blurring_step=blurring_step, radius=step_to_radius[blurring_step])
reconstructor.eval()
reconstructor.to("cuda")

subset = "text" # sanity text

if save_report:
    report_name = version + '_' + subset
    report_path = path_parts[:-1]
    report_path.append(report_name)
    report_path = os.path.join(*report_path)
    Path(report_path).mkdir(parents=True, exist_ok=True)


dataset = HDCDatasetTest(step=blurring_step, subset=subset)


psnrs = []
ssims = []
with torch.no_grad():
    for i in tqdm(range(len(dataset))):
    
        y, x, _ = dataset[i]
        y = y.to('cuda').unsqueeze(0)
        x = x.unsqueeze(0)

        # create reconstruction from observation
        with torch.no_grad():
            reco = reconstructor.forward(y)
        reco = reco.cpu().numpy()
        reco = np.clip(reco, 0, 1)

        # calculate quality metrics
        psnrs.append(PSNR(reco[0][0], x.numpy()[0][0]))
        ssims.append(SSIM(reco[0][0], x.numpy()[0][0]))


        if save_report:
            name = dataset.sharp_img_paths[i].split(".")[0].split("/")[-1]

            im = Image.fromarray(reco[0,0,:,:] * 255)
            im = im.convert("L")
            im.save(os.path.join(report_path,  name + "_" + str(i) + "_reco.png"))

            im = Image.fromarray(x.numpy()[0,0,:,:] * 255)
            im = im.convert("L")
            im.save(os.path.join(report_path, name + "_" + str(i) + "_gt.png"))

mean_psnr = np.mean(psnrs)
std_psnr = np.std(psnrs)
mean_ssim = np.mean(ssims)
std_ssim = np.std(ssims)

print('---')
print('Results:')
print('mean psnr: {:f}'.format(mean_psnr))
print('std psnr: {:f}'.format(std_psnr))
print('mean ssim: {:f}'.format(mean_ssim))
print('std ssim: {:f}'.format(std_ssim))

if save_report:
    report_dict = {'results': {'mean_psnr': float(np.mean(psnrs)) , 
                        'std_psnr': float(np.std(psnrs)),
                        'mean_ssim': float(np.mean(ssims)) ,
                        'std_ssim': float(np.std(ssims)) }}

    report_file_path =  os.path.join(report_path, 'report.yaml')
    with open(report_file_path, 'w') as file:
        documents = yaml.dump(report_dict, file)
