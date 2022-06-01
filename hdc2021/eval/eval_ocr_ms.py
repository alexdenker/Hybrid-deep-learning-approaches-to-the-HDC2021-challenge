"""
Eval OCR score of LGD on test images

"""
import os
from pathlib import Path

import torch
import yaml
from dival.util.plot import plot_images

import matplotlib.pyplot as plt 
from dival.measure import PSNR, SSIM
from tqdm import tqdm
import numpy as np 

import pytesseract
from fuzzywuzzy import fuzz

from skimage.transform import resize
from PIL import Image

from hdc2021.utils.blurred_dataset import BlurredDataModule, MultipleBlurredDataModule, HDCDatasetTest
from hdc2021.deblurrer.MultiScale_GDv4 import MultiScaleReconstructor


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
    print('True text (middle line): %s' % trueText[1])
    print(trueText, OCRtext, OCRGTtext)
    
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



print("Eval OCR ")
print("--------------------------------\n")

blurring_step = 4

step_to_radius = {4 : 0.015,
                  9 : 0.03,
                  14: 0.043,  #0.04
                  19: 0.0875}


save_report = True 
plot_examples = False 

base_path = "/localdata/AlexanderDenker/deblurring_experiments"
experiment_name = 'step_' + str(blurring_step)  
version = 'version_1'

path_parts = [base_path, 'multi_scale', experiment_name, 'default',
            version, 'checkpoints']
chkp_path = os.path.join(*path_parts)
hparams_file_path = os.path.join(os.path.join(*path_parts[:-1]), "hparams.yaml")

chkp_path = os.path.join(chkp_path, os.listdir(chkp_path)[0])
print(chkp_path)
print(hparams_file_path)
reconstructor = MultiScaleReconstructor.load_from_checkpoint(chkp_path, 
                                        hparams_file = hparams_file_path,
                                        blurring_step=blurring_step, 
                                        radius=step_to_radius[blurring_step])
reconstructor.eval()
reconstructor.to("cuda")



if save_report:
    report_name = version + '_'  + "_ocr"
    report_path = path_parts[:-1]
    report_path.append(report_name)
    report_path = os.path.join(*report_path)
    Path(report_path).mkdir(parents=True, exist_ok=True)

    report_dict = {}

psnrs = []
ssims = []
for font in ["Verdana", "Times"]:
    dataset = HDCDatasetTest(step=blurring_step, subset='text', font=font)


    ocr_acc = [] 
    ocri_acc = []
    with torch.no_grad():
        for i in tqdm(range(len(dataset))):
        
            y, x, text = dataset[i]
            y = y.to('cuda').unsqueeze(0)
            x = x.unsqueeze(0)

            # create reconstruction from observation
            with torch.no_grad():
                reco = reconstructor.forward(y)
            reco = reco.cpu().numpy()

            #fig, (ax1, ax2) = plt.subplots(1,2)
            #ax1.imshow(reco[0][0], cmap="gray")
            #ax2.imshow(x.numpy()[0][0], cmap="gray")
            #plt.show()

            reco = np.clip(reco, 0, 1)


            # calculate quality metrics
            psnrs.append(PSNR(reco[0][0], x.numpy()[0][0]))
            ssims.append(SSIM(reco[0][0], x.numpy()[0][0]))
            
            ocr_, ocri_ = evaluateImage(reco[0][0], x.numpy()[0][0], text)
            ocr_acc.append(ocr_)
            ocri_acc.append(ocri_)

            if i < 4:
                im = Image.fromarray(reco[0,0,:,:] * 255)
                im = im.convert("L")
                im.save(os.path.join(report_path, font + "_" + str(i) + "_reco.png"))

                im = Image.fromarray(x.numpy()[0,0,:,:] * 255)
                im = im.convert("L")
                im.save(os.path.join(report_path, font + "_" + str(i) + "_gt.png"))

    mean_ocr = np.mean(ocr_acc)
    std_ocr = np.std(ocr_acc)

    mean_ocri = np.mean(ocri_acc)
    std_ocri = np.std(ocri_acc)


    print('---')
    print('Results for {} :'.format(font))
    
    print('mean OCR acc: ', mean_ocr)
    print('std OCR acc: ', std_ocr)
    print('mean OCRI acc: ', mean_ocri)
    print('std OCRI acc: ', std_ocri)
    
    if save_report:
        report_dict[font] = {'results': {
                            'mean_ocr_acc': float(mean_ocr),
                            'std_ocr_acc': float(std_ocr),
                            'mean_ocri_acc': float(mean_ocri),
                            'std_ocri_acc': float(std_ocri) }}

mean_psnr = np.mean(psnrs)
std_psnr = np.std(psnrs)
mean_ssim = np.mean(ssims)
std_ssim = np.std(ssims)
print("BOTH: ")
print('mean psnr: {:f}'.format(mean_psnr))
print('std psnr: {:f}'.format(std_psnr))
print('mean ssim: {:f}'.format(mean_ssim))
print('std ssim: {:f}'.format(std_ssim))

if save_report:

    report_dict["full"] = {'results': { 
                            'mean_psnr': float(np.mean(psnrs)) , 
                            'std_psnr': float(np.std(psnrs)),
                            'mean_ssim': float(np.mean(ssims)) ,
                            'std_ssim': float(np.std(ssims))
    }}


    report_file_path =  os.path.join(report_path, 'report.yaml')
    with open(report_file_path, 'w') as file:
        documents = yaml.dump(report_dict, file)
