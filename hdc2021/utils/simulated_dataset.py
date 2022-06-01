"""
Generate new text images in the HDC2021 style. 

Image: 200x300px 

Adapted from: https://github.com/theophil-trippe/HDC_TUBerlin_version_2/blob/master/data_generator.py
"""


import torch 
from torchvision.transforms import ToTensor, Compose, CenterCrop, ToPILImage, RandomCrop

from sklearn.cluster import KMeans

import numpy as np 
import matplotlib.pyplot as plt 
from PIL import Image, ImageOps, ImageDraw, ImageFont, ImageFilter

import string
import random
import os 

from hdc2021.forward_model.blur import BlurOp


#VerdanaPath = os.path.join("/localdata/junickel/hdc2021/ocr", "verdanaRef.ttf")
#TimesPath = os.path.join("/localdata/junickel/hdc2021/ocr/", "times.ttf")
VerdanaPath = os.path.join("/home/adenker/deblurring/hdc_paper/hdc2021/ocr", "verdanaRef.ttf")
TimesPath = os.path.join("/home/adenker/deblurring/hdc_paper/hdc2021/ocr", "times.ttf")

def generate_background(size=(300, 200), intensity=215):
    """
    :param size: width x height of image to be generated
    :param intensity: background intensity
    :return: returns monochromatic grayscale PIL.Image
    """
    return Image.new('L', size, color=intensity)

def add_text(img, text=['Example'], positions=[(0, 0)], font_path=VerdanaPath, font_size=35, font_colour=0):
    """
    :param img: input image (PIL.Image)
    :param text: list containing the strings to be added
    :param positions: list containing the positions where the text should be placed on the input image
    :param font_path: path pointing to the location of the font to be used
    :param font_size:
    :param font_colour:
    :return: input image with text added onto it
    """
    font = ImageFont.truetype(font_path, font_size, layout_engine=ImageFont.LAYOUT_RAQM)

    assert len(text) == len(positions)
    d = ImageDraw.Draw(img)
    for l in range(len(text)):
        d.text(positions[l], text[l], font=font, fill=font_colour)
    return img, text


def generate_text(num_lines=3, line_len_min=10, line_len_max=10):
    """
    :param num_lines: number of lines to be returned
    :param line_len_min: minimal length each line should have
    :param line_len_max: maximal lentgh each line should have
    :return: list of length 'num_lines' containing random strings
    """
    alphabet = [string.ascii_lowercase + ' ', string.ascii_uppercase + ' ']
    alphabet_len = len(alphabet[0])

    characters = 3 * string.ascii_letters + \
                 2 * string.digits + \
                 ' +!#&()*+-/:;=>?\^'
    num_characters = len(characters)

    out = []
    for l in range(num_lines):
        length = random.randint(line_len_min, line_len_max)
        str_l = ''
        for c in range(length):
            char_id = random.randint(0, num_characters-1)  # determines which character to choose
            str_l = str_l + characters[char_id]
        out.append(str_l)

    return out


def gen_and_add_text_HDC_style(img, font_size=35, font_path=VerdanaPath, line_spacing=5, font_colour=20):
    """
    :param img: input image (PIL.Image)
    :param font_size: integer
    :param font_path: pick one of the font paths specified in config
    :param line_spacing: space between lines
    :param font_colour: uint8 integer determining the font colour
    :return: randomly generates three string-lines and adds them to the input image in HDC-style
    """
    cwd = os.getcwd()
    font = ImageFont.truetype(font_path, font_size)
    text = generate_text(3, 10, 10)
    d = ImageDraw.Draw(img)
    positions = []
    text_sizes = []
    for i in range(3):
        w, h = d.textsize(text[i], font=font)
        text_sizes.append((w, h))
        W, H = (img.width - w) / 2, (img.height - h) / 2 + 5
        if len(positions) == 1:
            H = H + line_spacing + font_size
        if len(positions) == 2:
            H = H - (line_spacing + font_size)
        positions.append((W, H))
    text = [text[2], text[0], text[1]]
    positions = [positions[2], positions[0], positions[1]]
    return add_text(img, text, positions, font_path, font_size, font_colour)


def add_gaussian_noise(img, mean=0., var=30.):
    """
    :param img: input image (PIL.Image)
    :param mean: should stay 0 unless image should be brightened up or darkened
    :param var: determines amount of noise to be added by this funciton
    :return: input image with random gaussian noise added
    """
    gaussian = np.random.normal(mean, var**0.5, (img.height, img.width))
    #poisson = np.random.poisson(np.asarray(img) / 255.0 * PEAK) / PEAK * 255
    return Image.fromarray(np.clip(img + gaussian, 0, 255).astype(np.uint8))




def generate_hdc_image(font, font_colour=34, font_size=31, intensity=212): 
    """
    Generate a new image in the style of HDC2021

    """
    if font == 'Times':
        font_path = TimesPath
    else:
        font_path = VerdanaPath


    img = generate_background(intensity=intensity)
    img, text = gen_and_add_text_HDC_style(img, font_size=font_size,font_colour=font_colour, font_path=font_path)
    
    img = img.point(lambda p: font_colour if p < intensity  else intensity)
    img = img.resize((2360, 1460), resample=Image.NEAREST)
    #img = add_gaussian_noise(img)

    sharp = ToTensor()(img)

    return sharp, text


class SimulatedDataset(torch.utils.data.Dataset):
    """
    Generates a dataset containing synthesized data.
    WARNING: two calls of getitem produce seperate samples
    """
    def __init__(self,subset,length=15,device=None,sample_generator=generate_hdc_image, blurring_step=None):
        """
        :param step: blurriness
        :param length: legth of dataset (needed to define __len__)
        :param device: device to locate the samples on
        """

        self.device = device
        self.length = length
        self.sample_generator = sample_generator
        self.subset = subset
        self.blurring_step = blurring_step

        if self.blurring_step is not None:
            self.blur = BlurOp(inp_size=[1460, 2360], kernel_size=701)

            self.blur.load_state_dict(torch.load('forward_model/weights/step_' + str(blurring_step) + ".pt"))
            self.blur.conv2d.weight.data = self.blur.conv2d.weight.data / torch.sum(self.blur.conv2d.weight.data)
            self.blur.eval()


    def __len__(self):
        if self.subset == "val":
            return 10
        else:
            return self.length

    def __getitem__(self, IDX):
        with torch.no_grad():
            font = random.choice(["Times", "Verdana"])
            sharp, txt = self.sample_generator(font)

            if self.blurring_step is not None:
                PEAK = 8
                sharp = torch.poisson(sharp * PEAK * 255) / (PEAK * 255.)

                blurry = self.blur(torch.unsqueeze(sharp, dim=0))
                blurry = torch.poisson(blurry * 2* 255) / (255.*2)

                out = [torch.squeeze(blurry, dim=0), sharp]
            else:
                out = [sharp]

            if self.device is not None:
                out = [x.to(self.device) for x in out]

            #if self.rel_noise_var != 0:
            #    out[0] = add_relative_noise(out[0], variance=self.rel_noise_var).float()

            return tuple(out) + (txt,)


class Urban100Dataset(torch.utils.data.Dataset):
    def __init__(self,subset,length=15,device=None, blurring_step=None, base_path="/home/adenker/deblurring/hdc_paper/hdc2021/utils/urban100"):
        """
        :param step: blurriness
        :param length: legth of dataset (needed to define __len__)
        :param device: device to locate the samples on
        """
        self.base_path = base_path
        self.device = device
        self.length = length
        self.subset = subset
        self.blurring_step = blurring_step

        self.img_names = os.listdir(self.base_path)

        if self.blurring_step is not None:
            self.blur = BlurOp(inp_size=[1460, 2360], kernel_size=701)

            self.blur.load_state_dict(torch.load('forward_model/weights/step_' + str(blurring_step) + ".pt"))
            self.blur.conv2d.weight.data = self.blur.conv2d.weight.data / torch.sum(self.blur.conv2d.weight.data)
            self.blur.eval()


    def __len__(self):
        if self.subset == "val":
            return 10
        else:
            return self.length

    def __getitem__(self, IDX):

        def load(path):
            pre_trans = Compose([ToTensor(), RandomCrop((200,300))])

            with Image.open(path, 'r') as file:
                img = pre_trans(file.convert('L'))
            return img

        with torch.no_grad():

            img_name = np.random.choice(self.img_names)
            img = load(os.path.join(self.base_path, img_name))
            binary_image = KMeans(n_clusters=2, random_state=42,verbose=False,n_init=1,tol=1e-3).fit_predict(img.flatten().reshape(-1, 1))

            color1 = np.random.rand()*4 + 210
            color2 = np.random.rand()*4 + 32

            txt = ""

            binary_image = binary_image.reshape(200, 300)*(color1/255. - color2/255.) + color2/255.
            img = torch.from_numpy(binary_image).float().unsqueeze(0).unsqueeze(0)

            sharp = torch.nn.functional.interpolate(img, size=(1460, 2360), mode="nearest").squeeze(0)

            if self.blurring_step is not None:
                PEAK = 8
                sharp = torch.poisson(sharp * PEAK * 255) / (PEAK * 255.)

                blurry = self.blur(torch.unsqueeze(sharp, dim=0))
                blurry = torch.poisson(blurry * 2* 255) / (255.*2)

                out = [torch.squeeze(blurry, dim=0), sharp]
            else:
                out = [sharp]

            if self.device is not None:
                out = [x.to(self.device) for x in out]

            #if self.rel_noise_var != 0:
            #    out[0] = add_relative_noise(out[0], variance=self.rel_noise_var).float()

            return tuple(out) + (txt,)
