"""
PSFDataset and HDCDataset from https://github.com/theophil-trippe/HDC_TUBerlin_version_1

"""

import os 
import pytorch_lightning as pl 

import torch
from torch.utils.data import DataLoader, TensorDataset, Dataset
import numpy as np 
from torchvision import transforms

from hdc2021.utils import data_util
from hdc2021.utils.simulated_dataset import SimulatedDataset, Urban100Dataset

import torchvision 
import torchvision.transforms as transforms

import matplotlib.pyplot as plt 
import torch 
from torchvision.transforms import ToTensor, CenterCrop, ToPILImage, Compose, RandomCrop
from PIL import Image, ImageOps
import os 
import numpy as np
import random
from os.path import join


DATA_PATH = "/localdata/helsinki_deblur/"


def change_im_range(img_tens,
                    new_min,
                    new_max):
    old_min = torch.min(img_tens).item()
    old_max = torch.max(img_tens).item()
    ret = ((img_tens - old_min) * ((new_max - new_min) / (old_max - old_min))) + new_min
    return ret

# creates a torch.dataset for the PSF and LSF data (3 samples per step)
class PSFDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        step,
        transform=None, 
        shift_bg=None,
    ):
        self.type = "PSF"
        self.transform = transform
        self.step = step
        self.sample_ids = ['LSF_X', 'LSF_Y', 'PSF']
        if shift_bg is not None:
            self.bg = shift_bg(step)
        else:
            self.bg = None
        font = 'Times'
        # choose directory according to step and (either) font
        path = join(DATA_PATH, 'step' + str(step), font)

        # load file names with location
        self.feature_paths = [join(path, 'CAM02', 'focusStep_' + str(step) + '_' + idx + '.tif') for idx in self.sample_ids]
        self.target_paths = [join(path, 'CAM01', 'focusStep_' + str(step) + '_' + idx + '.tif') for idx in self.sample_ids]

        assert len(self.feature_paths) == len(self.target_paths)

    def __len__(self):
        return len(self.feature_paths)

    def __getitem__(self, idx):

        # load sample from storage
        def load(paths, IDX):
            pre_trans= Compose([ToTensor()])
            with Image.open(paths[IDX], 'r') as file:
                img = pre_trans(file.point(lambda i: i * (1. / 256)).convert('L'))
            return img

        out = [load(self.feature_paths, idx),  # loads feature
               load(self.target_paths, idx)]  # loads target

        if self.bg is not None:
            out = list(torch.chunk(Shift(self.bg)(torch.cat((out[0], out[1]), 0)),
                                   2))  # torch.cat -> transform -> torch.chunk necessary for random transforms

        blur_min = torch.min(out[0]).item()
        blur_max = torch.max(out[0]).item()
        shar_min = torch.min(out[1]).item()
        shar_max = torch.max(out[1]).item()

        out[0] = change_im_range(out[0], new_min=0.0, new_max=1.0)
        out[1] = change_im_range(out[1], new_min=0.0, new_max=1.0)

        if self.transform is not None:
            out = [self.transform(x) for x in out]

        out[0] = change_im_range(out[0], new_min=blur_min, new_max=blur_max)
        out[1] = change_im_range(out[1], new_min=shar_min, new_max=shar_max)

        return tuple(out) + (self.sample_ids[idx],)
    
class Shift(object):
    """ Shift. """

    def __init__(self, shift):
        self.shift = shift

    def __call__(self, inputs):
        return inputs - self.shift

    
    
# creates a torch.dataset for the font data
class HDCDataset(torch.utils.data.Dataset):
    def __init__(self,step,subset,font=None,transform=None,shift_bg=None):
        self.transform = transform
        self.subset = subset
        if shift_bg is not None:
            self.bg = shift_bg(step)
        else:
            self.bg = None
        # create validation split, there are 100 images per font. We use the last 10 for validation
        if subset == "val":
            self.sample_ids = [str(i) for i in list(range(91, 101))]
        elif subset == "train":
            self.sample_ids = [str(i) for i in list(range(1, 91))]
        else:
            print("ERROR: subset need to be one of 'val' or 'train' ")

        # set the font_id in order to set the right filenames later on
        font_names = {'Verdana': 'verdanaRef',
                      'Times': 'timesR'}
        if font == 'Verdana':
            del font_names['Times']
        elif font == 'Times':
            del font_names['Verdana']
        

        self.blury_img_paths = []
        self.sharp_img_paths = []
        self.text_target_paths = []
        # choose directory according to step and font
        for font in font_names:
            path = join(DATA_PATH, 'step' + str(step), font.capitalize())
            font_id = font_names[font]
            # load data files
            def sample_name(sample_id):
                return 'focusStep_' + str(step) + '_' + font_id + '_size_30_sample_' + str(sample_id).zfill(4)
            blury_img_paths = [join(path, 'CAM02', sample_name(sample_id) + '.tif') for sample_id in self.sample_ids]
            sharp_img_paths = [join(path, 'CAM01', sample_name(sample_id) + '.tif') for sample_id in self.sample_ids]
            text_target_paths = [join(path, 'CAM01', sample_name(sample_id) + '.txt') for sample_id in self.sample_ids]

            self.blury_img_paths.extend(blury_img_paths)
            self.sharp_img_paths.extend(sharp_img_paths)
            self.text_target_paths.extend(text_target_paths)

        assert len(self.blury_img_paths) == len(self.sharp_img_paths)
        assert len(self.blury_img_paths) == len(self.text_target_paths)

    def __len__(self):
        return len(self.blury_img_paths)

    def __getitem__(self, idx):
        # create the text target.
        with open(self.text_target_paths[idx], 'r') as f:
            text_target = f.readlines()
        # if self.subset == 'val':  # skip this step if not for validation set, to speed up DataLoading
            text_target = [text.rstrip() for text in text_target]

        # load sample from storage
        def load(paths, IDX):
            # pre_trans = Compose([CenterCrop((1456, 2352)), ToTensor()])
            pre_trans = Compose([ToTensor()])

            with Image.open(paths[IDX], 'r') as file:
                img = pre_trans(file.point(lambda i: i * (1. / 256)).convert('L'))
            return img

        out = [load(self.blury_img_paths, idx), # loads feature
               load(self.sharp_img_paths, idx)] # loads target

        if self.bg is not None:
            out = list(torch.chunk(Shift(self.bg)(torch.cat((out[0], out[1]), 0)),
                                   2))  # torch.cat -> transform -> torch.chunk necessary for random transforms

        blur_min = torch.min(out[0]).item()
        blur_max = torch.max(out[0]).item()
        shar_min = torch.min(out[1]).item()
        shar_max = torch.max(out[1]).item()

        out[0] = change_im_range(out[0], new_min=0.0, new_max=1.0)
        out[1] = change_im_range(out[1], new_min=0.0, new_max=1.0)

        if self.transform is not None:
            out = [self.transform(x) for x in out]

        out[0] = change_im_range(out[0], new_min=blur_min, new_max=blur_max)
        out[1] = change_im_range(out[1], new_min=shar_min, new_max=shar_max)

        return tuple(out) + (text_target,)


# creates a torch.dataset for the font data
class HDCDatasetTest(torch.utils.data.Dataset):
    def __init__(self, step, subset, transform=None, shift_bg=None, font=None, base_path = "/localdata/helsinki_deblur_test/"):
        """
        subset: "text", "sanity"
            "text": 40 images of text (20 Times, 20 Verdana)
            "sanity": 16 natural images 
        font:  optional, only matters for subset = "text"
             "Verdana", "Times"
        """
        assert subset in ['text', 'sanity'], "subset has to be either test or sanity"

        self.step = step 
        self.transform = transform
        self.subset = subset
        self.font = font
        self.base_path = base_path

        if shift_bg is not None:
            self.bg = shift_bg(step)
        else:
            self.bg = None

        self.blury_img_paths = []
        self.sharp_img_paths = []
        self.text_target_paths = []

        font_names = {'Verdana': 'verdanaRef',
                            'Times': 'timesR'}
        if font == 'Verdana':
            del font_names['Times']
        elif font == 'Times':
            del font_names['Verdana']


        if 0 <= self.step <= 4:
            step_folder = "steps_0_to_4"
        elif 5 <= self.step <= 9:
            step_folder = "steps_5_to_9"
        elif 10 <= self.step <= 14:
            step_folder = "steps_10_to_14"
        elif 15 <= self.step <= 19:
            step_folder = "steps_15_to_19"

        if self.subset == "text":
            # choose directory according to step and font
            for font in font_names:
            
                path = join(self.base_path, step_folder)
                font_id = font_names[font]
                # load data files
                def sample_name(sample_id):
                    return 'focusStep_' + str(step) + '_' + font_id + '_size_30_sample_' + str(sample_id).zfill(4)
                blury_img_paths = [join(path, 'CAM02_blurred', sample_name(sample_id) + '.tif') for sample_id in range(1,21)]
                sharp_img_paths = [join(path, 'CAM01_focused', sample_name(sample_id) + '.tif') for sample_id in range(1,21)]
                text_target_paths = [join(path, 'CAM01_focused', sample_name(sample_id) + '.txt') for sample_id in range(1,21)]

                self.blury_img_paths.extend(blury_img_paths)
                self.sharp_img_paths.extend(sharp_img_paths)
                self.text_target_paths.extend(text_target_paths)

        elif self.subset == "sanity":
            sanity_img_names = ["Image_Green", "Image_Green_10_200", "Image_Green_12_200", "Image_bird", "Image_blackbird2_200", 
                                "Image_dandelion_200", "Image_deer", "Image_discgolf_200", "Image_ipanema", "Image_moon", "Image_naakka_200",
                                "Image_skyline_200", "Image_squirrel_200", "Image_swanset", "Image_twigs_200", "QRcode"]
            
            
            path = join(self.base_path, step_folder)

            def sample_name(img_name):
                    return 'focusStep_' + str(step) + '_' + img_name
            blury_img_paths = [join(path, 'CAM02_blurred', sample_name(name) + '.tif') for name in sanity_img_names]
            sharp_img_paths = [join(path, 'CAM01_focused', sample_name(name) + '.tif') for name in sanity_img_names]

            self.blury_img_paths.extend(blury_img_paths)
            self.sharp_img_paths.extend(sharp_img_paths)

        assert len(self.blury_img_paths) == len(self.sharp_img_paths)

    def __len__(self):
        return len(self.blury_img_paths)

    def __getitem__(self, idx):
        # create the text target, if subset = "text"
        if self.subset == "text":
            with open(self.text_target_paths[idx], 'r') as f:
                text_target = f.readlines()
            # if self.subset == 'val':  # skip this step if not for validation set, to speed up DataLoading
                text_target = [text.rstrip() for text in text_target]
        else:
            text_target = ""
        # load sample from storage
        def load(paths, IDX):
            # pre_trans = Compose([CenterCrop((1456, 2352)), ToTensor()])
            pre_trans = Compose([ToTensor()])

            with Image.open(paths[IDX], 'r') as file:
                img = pre_trans(file.point(lambda i: i * (1. / 256)).convert('L'))
            return img

        out = [load(self.blury_img_paths, idx), # loads feature
               load(self.sharp_img_paths, idx)] # loads target

        if self.bg is not None:
            out = list(torch.chunk(Shift(self.bg)(torch.cat((out[0], out[1]), 0)),
                                   2))  # torch.cat -> transform -> torch.chunk necessary for random transforms

        blur_min = torch.min(out[0]).item()
        blur_max = torch.max(out[0]).item()
        shar_min = torch.min(out[1]).item()
        shar_max = torch.max(out[1]).item()

        out[0] = change_im_range(out[0], new_min=0.0, new_max=1.0)
        out[1] = change_im_range(out[1], new_min=0.0, new_max=1.0)

        if self.transform is not None:
            out = [self.transform(x) for x in out]

        out[0] = change_im_range(out[0], new_min=blur_min, new_max=blur_max)
        out[1] = change_im_range(out[1], new_min=shar_min, new_max=shar_max)

        return tuple(out) + (text_target,)


# ---- default background: ----
def _psf_shift(step):
    return (torch.cat(PSFDataset(step)[2][0:-1], 0))


class BlurredDataModule(pl.LightningDataModule):
    def __init__(self, batch_size:int = 4, blurring_step:int=0, num_data_loader_workers:int=8, shift_bg=False):
        """
        Create a PyTorchLightning DataModule.

        Parameters:
            shift_bg: optional,
                False, dont substract background from images
                True, substract background (background = PSF from technical image)
            # In general, this should be FALSE for training the deblurring network and TRUE for training the blurring network
        """
        super().__init__()

        assert blurring_step in np.arange(20), "blurring_step has to an integer between 0 and 19"

        self.batch_size = batch_size    
        self.blurring_step = blurring_step
        self.num_data_loader_workers = num_data_loader_workers
        self.shift_bg = shift_bg 

    def prepare_data(self):
        None 

    def setup(self, stage:str = None): 

        shift_bg = _psf_shift if self.shift_bg == True else None

        # Assign train/val datasets for use in dataloaders
        if stage == 'fit' or stage is None:
            # load training data 

            self.blurred_dataset_train = HDCDataset(step=self.blurring_step, subset='train', shift_bg=shift_bg)
            self.dims = tuple(self.blurred_dataset_train[0][0].shape)

            self.blurred_dataset_validation = HDCDataset(step=self.blurring_step, subset='val', shift_bg=shift_bg)

        # Assign test dataset for use in dataloader(s)
        if stage == 'test' or stage is None:
            self.blurred_dataset_test =  HDCDatasetTest(step=self.blurring_step, subset='text', shift_bg=shift_bg)

            self.dims = tuple(self.blurred_dataset_test[0][0].shape)

    def train_dataloader(self):
            """
            Data loader for the training data.

            Returns
            -------
            DataLoader
                Training data loader.

            """
            return DataLoader(self.blurred_dataset_train, batch_size=self.batch_size,
                            num_workers=self.num_data_loader_workers,
                            shuffle=True, pin_memory=True)

    def val_dataloader(self):
        """
        Data loader for the validation data.

        Returns
        -------
        DataLoader
            Validation data loader.

        """
        return DataLoader(self.blurred_dataset_validation, batch_size=self.batch_size,
                          num_workers=self.num_data_loader_workers,
                          shuffle=False, pin_memory=True)

    def test_dataloader(self):
        """
        Data loader for the test data.

        Returns
        -------
        DataLoader
            Test data loader.

        """
        return DataLoader(self.blurred_dataset_test, batch_size=self.batch_size,
                          num_workers=self.num_data_loader_workers,
                          shuffle=False, pin_memory=True)


class SimulatedDataModule(pl.LightningDataModule):
    def __init__(self, batch_size:int = 4, blurring_step:int=0, num_data_loader_workers:int=8):
        """
        Create a PyTorchLightning DataModule.

        Parameters:
            shift_bg: optional,
                False, dont substract background from images
                True, substract background (background = PSF from technical image)
            # In general, this should be FALSE for training the deblurring network and TRUE for training the blurring network
        """
        super().__init__()

        assert blurring_step in np.arange(20), "blurring_step has to an integer between 0 and 19"

        self.batch_size = batch_size    
        self.blurring_step = blurring_step
        self.num_data_loader_workers = num_data_loader_workers

    def prepare_data(self):
        None 

    def setup(self, stage:str = None): 


        # Assign train/val datasets for use in dataloaders
        if stage == 'fit' or stage is None:
            # load training data 

            self.simulated_dataset_train = SimulatedDataset(subset='train', length=200,blurring_step=self.blurring_step)
            #self.dims = tuple(self.blurred_dataset_train[0][0].shape)

            self.simulated_dataset_val = SimulatedDataset(subset='val',blurring_step=self.blurring_step)

        # Assign test dataset for use in dataloader(s)
        if stage == 'test' or stage is None:
            self.simulated_dataset_test = SimulatedDataset(subset='val',blurring_step=self.blurring_step)

            #self.dims = tuple(self.blurred_dataset_test[0][0].shape)

    def train_dataloader(self):
            """
            Data loader for the training data.

            Returns
            -------
            DataLoader
                Training data loader.

            """
            return DataLoader(self.simulated_dataset_train, batch_size=self.batch_size,
                            num_workers=self.num_data_loader_workers,
                            shuffle=True, pin_memory=True)

    def val_dataloader(self):
        """
        Data loader for the validation data.

        Returns
        -------
        DataLoader
            Validation data loader.

        """
        return DataLoader(self.simulated_dataset_val, batch_size=self.batch_size,
                          num_workers=self.num_data_loader_workers,
                          shuffle=False, pin_memory=True)

    def test_dataloader(self):
        """
        Data loader for the test data.

        Returns
        -------
        DataLoader
            Test data loader.

        """
        return DataLoader(self.simulated_dataset_test, batch_size=self.batch_size,
                          num_workers=self.num_data_loader_workers,
                          shuffle=False, pin_memory=True)



class MultipleBlurredDataModule(BlurredDataModule):
    def __init__(self, batch_size:int = 4, blurring_step:int=0, num_data_loader_workers:int=8):
        """
        Create a PyTorchLightning DataModule.

        Parameters:
            shift_bg: optional,
                False, dont substract background from images
                True, substract background (background = PSF from technical image)
            # In general, this should be FALSE for training the deblurring network and TRUE for training the blurring network
        """
        super().__init__()

        assert blurring_step in np.arange(20), "blurring_step has to an integer between 0 and 19"

        self.batch_size = batch_size    
        self.blurring_step = blurring_step
        self.num_data_loader_workers = num_data_loader_workers

    def setup(self, stage:str = None): 
        
        # Assign train/val datasets for use in dataloaders
        if stage == 'fit' or stage is None:
            # load training data 

            self.blurred_dataset_train = HDCDataset(step=self.blurring_step, subset='train', shift_bg=None)
            self.simulated_dataset_train = SimulatedDataset(subset='train', length=200,blurring_step=self.blurring_step)
            self.urban100_dataset_train = Urban100Dataset(subset='train', length=200, blurring_step=self.blurring_step)
            self.dims = tuple(self.blurred_dataset_train[0][0].shape)

            self.blurred_dataset_validation = HDCDataset(step=self.blurring_step, subset='val', shift_bg=None)
            self.simulated_dataset_validation = SimulatedDataset(subset='val', length=10,blurring_step=self.blurring_step)

            #self.simulated_dataset_val = SimulatedDataset(subset='val')

        # Assign test dataset for use in dataloader(s)
        if stage == 'test' or stage is None:
            self.blurred_dataset_test = HDCDatasetTest(step=self.blurring_step, subset='text', shift_bg=None)

            self.dims = tuple(self.blurred_dataset_test[0][0].shape)
    
    def train_dataloader(self):
        """
        Data loader for the training data.

        Returns
        -------
        DataLoader
            Training data loader.

        """
        loaders = {'hdc_loader' : DataLoader(self.blurred_dataset_train, batch_size=self.batch_size,
                        num_workers=self.num_data_loader_workers, shuffle=True, pin_memory=True),
                    'sim_loader': DataLoader(self.simulated_dataset_train, batch_size=self.batch_size,
                        num_workers=self.num_data_loader_workers, shuffle=False, pin_memory=False),
                    'urban100_loader': DataLoader(self.urban100_dataset_train, batch_size=self.batch_size,
                        num_workers=self.num_data_loader_workers, shuffle=False, pin_memory=False)
                        }
        return loaders

    def val_dataloader(self):
        """
        Data loader for the validation data.

        Returns
        -------
        DataLoader
            Validation data loader.

        """
        
        return DataLoader(self.blurred_dataset_validation, batch_size=self.batch_size,
                            num_workers=self.num_data_loader_workers, shuffle=False, pin_memory=True)


    def test_dataloader(self):
        """
        Data loader for the test data.

        Returns
        -------
        DataLoader
            Test data loader.

        """
        return DataLoader(self.blurred_dataset_test, batch_size=self.batch_size,
                          num_workers=self.num_data_loader_workers,
                          shuffle=False, pin_memory=True)



if __name__ == "__main__":
    import matplotlib.pyplot as plt 
    from torchvision.transforms import Compose, CenterCrop, ToTensor, ToPILImage, Resize


    dataset = HDCDataset(step=6, subset="train")
    print(len(dataset))

    dataset = HDCDataset(step=6, subset="val")
    print(len(dataset))

    for i in range(len(dataset)):
        y, x, text = dataset[i]
        print(y.shape, x.shape)
        print(text)
        fig, (ax1, ax2) = plt.subplots(1,2)

        ax1.imshow(y[0,:,:], cmap="gray")
        ax1.set_title(text)
        ax2.imshow(x[0,:,:], cmap="gray")
        plt.show()
    

    """
    dataset = HDCDatasetTest(step=6, subset="text",font="Verdana")
    print(len(dataset))

    for i in range(len(dataset)):
        y, x, text = dataset[i]
        print(y.shape, x.shape)
        print(text)
        fig, (ax1, ax2) = plt.subplots(1,2)

        ax1.imshow(y[0,:,:], cmap="gray")
        ax1.set_title(text)
        ax2.imshow(x[0,:,:], cmap="gray")
        plt.show()
    """

    """
    dataset = BlurredDataModule(blurring_step=13, shift_bg=True)
    dataset.prepare_data()
    dataset.setup()

    for batch in dataset.train_dataloader():
        y, x, text = batch 
        print(x.shape, y.shape)
        print(text)
        fig, (ax1, ax2) = plt.subplots(1,2)

        ax1.imshow(y[0,0,:,:], cmap="gray")

        ax2.imshow(x[0,0,:,:], cmap="gray")
        plt.show()
        #print(dataset.blurred_dataset_train.bg.shape)
        
        #pred_img = ToPILImage(mode='L')(x[0,0,:,:])
        #pred_img.save("test.png", format="png")
        break
    """

    """
    dataset = MultipleBlurredDataModule(blurring_step=8)
    dataset.prepare_data()
    dataset.setup()

    for batch in dataset.train_dataloader()['sim_loader']:
        x_sim, text = batch

        break 

    for batch in dataset.train_dataloader()['hdc_loader']:
        y, x_meas, text = batch

        break 

    fig, (ax1, ax2) = plt.subplots(1,2)

    ax1.imshow(x_sim[0,0,:,:], cmap="gray")
    ax1.set_title("simulated data")
    ax2.imshow(x_meas[0,0,:,:], cmap="gray")
    ax2.set_title("real HDC2021 data")
    plt.show()
    #print(dataset.blurred_dataset_train.bg.shape)
    
    #pred_img = ToPILImage(mode='L')(x[0,0,:,:])
    #pred_img.save("test.png", format="png")
    """