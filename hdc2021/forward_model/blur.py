"""
Combined blur and distortion model. 
Adapted from: https://github.com/theophil-trippe/HDC_TUBerlin_version_1

"""

from functools import partial

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as f

import numpy as np 


def fft_conv_nd(signal: Tensor, kernel: Tensor, bias: Tensor = None, padding: int = 0, stride: int = 1,
                padding_mode: str = "zeros") -> Tensor:
    """Performs N-d convolution of Tensors using a fast fourier transform, which is very fast for large kernel sizes.
    Also, optionally adds a bias Tensor after the convolution (in order ot mimic the PyTorch direct convolution).
    :param signal: Input tensor to be convolved with the kernel.  Shape: (batch, nchan, nsamples)
    :param kernel: Convolution kernel.  Shape: (channels_out, channels_in, kernel_size)
    :param bias: (Optional) bias tensor to add to the output.  Shape:  (channels_out, )
    :param padding: Number of zero samples to pad the input on the last dimension.
    :param stride: Convolution stride length
    :return: Convolved tensor
    """
    ndims = len(signal.shape)
    conv_dims = ndims - 2

    # Pad the input signal & kernel tensors
    signal_padding = conv_dims * [padding, padding]
    signal = f.pad(signal, signal_padding, mode=padding_mode)
    kernel_padding = torch.tensor(
        [[0, signal.size(i) - kernel.size(i)] for i in range(ndims - 1, 1, -1)]).flatten().tolist()
    padded = f.pad(kernel, kernel_padding)
    

    # # Perform fourier convolution -- FFT, matrix multiply, then IFFT
    # signal_fr = torch.rfft(signal, conv_dims)
    # padded_fr = torch.rfft(padded, conv_dims)
    # output_fr = complex_matmul(signal_fr, padded_fr)
    # signal_sizes = [signal.size(i) for i in range(2, ndims)]
    # output = torch.irfft(output_fr, conv_dims, signal_sizes=signal_sizes)

    # less memory usage:
    # Perform fourier convolution -- FFT, matrix multiply, then IFFT
    output = torch.fft.rfft2(signal)#, conv_dims)
    padded_fr = torch.fft.rfft2(padded) #, conv_dims)
    output = torch.multiply(padded_fr,output) #complex_matmul(output, padded_fr)
    signal_sizes = [signal.size(i) for i in range(2, ndims)]
    output = torch.fft.irfft2(output)#, s=signal_sizes)#, conv_dims, s=signal_sizes)

    # Keep outputs at strided intervals, then remove extra padded values
    stride_slices = [slice(0, output.shape[0]), slice(0, output.shape[1])] + \
                    [slice(0, output.shape[i], stride) for i in range(2, ndims)]
    #crop_slices = [slice(0, output.shape[0]), slice(0, output.shape[1])] + \
    #              [slice(0, (signal.size(i) - kernel.size(i)) // stride + 1) for i in range(2, ndims)]
    crop_slices = [slice(0, output.shape[0]), slice(0, output.shape[1])] + \
                  [slice(kernel.size(i) - 1, signal.size(i)) for i in range(2, ndims)]
    #output = output[stride_slices]
    output = output[crop_slices].contiguous()

    # Optionally, add a bias term before returning.
    if bias is not None:
        bias_shape = tuple([1, -1] + conv_dims * [1])
        output += bias.view(bias_shape)

    return output



def wiener_filter_torch(signal: Tensor, kernel: Tensor, padding: int = 0, stride: int = 1,
                padding_mode: str = "zeros") -> Tensor:
    """Performs N-d convolution of Tensors using a fast fourier transform, which is very fast for large kernel sizes.
    Also, optionally adds a bias Tensor after the convolution (in order ot mimic the PyTorch direct convolution).
    :param signal: Input tensor to be convolved with the kernel.  Shape: (batch, nchan, nsamples)
    :param kernel: Convolution kernel.  Shape: (channels_out, channels_in, kernel_size)
    :param bias: (Optional) bias tensor to add to the output.  Shape:  (channels_out, )
    :param padding: Number of zero samples to pad the input on the last dimension.
    :param stride: Convolution stride length
    :return: Convolved tensor
    """
    ndims = len(signal.shape)
    conv_dims = ndims - 2

    # Pad the input signal & kernel tensors
    signal_padding = conv_dims * [padding, padding]
    signal = f.pad(signal, signal_padding, mode=padding_mode)
    kernel_padding = torch.tensor(
        [[0, signal.size(i) - kernel.size(i)] for i in range(ndims - 1, 1, -1)]).flatten().tolist()
    padded = f.pad(kernel, kernel_padding)
    

    # # Perform fourier convolution -- FFT, matrix multiply, then IFFT
    # signal_fr = torch.rfft(signal, conv_dims)
    # padded_fr = torch.rfft(padded, conv_dims)
    # output_fr = complex_matmul(signal_fr, padded_fr)
    # signal_sizes = [signal.size(i) for i in range(2, ndims)]
    # output = torch.irfft(output_fr, conv_dims, signal_sizes=signal_sizes)

    # less memory usage:
    # Perform fourier convolution -- FFT, matrix multiply, then IFFT
    output = torch.fft.rfft2(signal)#, conv_dims)
    padded_fr = torch.fft.rfft2(padded) #, conv_dims)

    w_filter = torch.conj(padded_fr)/(padded_fr**2 + 1e-2)

    output = torch.multiply(w_filter,output) #complex_matmul(output, padded_fr)
    signal_sizes = [signal.size(i) for i in range(2, ndims)]
    output = torch.fft.irfft2(output)#, s=signal_sizes)#, conv_dims, s=signal_sizes)

    # Keep outputs at strided intervals, then remove extra padded values
    stride_slices = [slice(0, output.shape[0]), slice(0, output.shape[1])] + \
                    [slice(0, output.shape[i], stride) for i in range(2, ndims)]
    #crop_slices = [slice(0, output.shape[0]), slice(0, output.shape[1])] + \
    #              [slice(0, (signal.size(i) - kernel.size(i)) // stride + 1) for i in range(2, ndims)]
    crop_slices = [slice(0, output.shape[0]), slice(0, output.shape[1])] + \
                  [slice(kernel.size(i) - 1, signal.size(i)) for i in range(2, ndims)]
    #output = output[stride_slices]
    output = output[crop_slices].contiguous()


    return output




class _FFTConv(nn.Module):
    """Base class for PyTorch FFT convolution layers."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, padding: int = 0, stride: int = 1,
                 bias: bool = True, padding_mode: str = "zeros"):
        """
        :param in_channels: Number of channels in input tensors
        :param out_channels: Number of channels in output tensors
        :param kernel_size: Size of the 2D convolution kernel.  (i.e. kernel_size=3 gives a 3x3 kernel)
        :param padding: Amount of zero-padding to add to the input tensor
        :param stride: Convolution stride length. Defaults to 1, as in standard convolution
        :param bias: If True, includes an additional bias term, which is added to the output after convolution
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.padding_mode = padding_mode
        self.stride = stride
        self.use_bias = bias

        self.weight = None
        self.bias = None

    def forward(self, signal):
        return fft_conv_nd(signal, self.weight, bias=self.bias, padding=self.padding, stride=self.stride,
                           padding_mode=self.padding_mode)


class FFTConv2d(_FFTConv):
    """PyTorch 2D convoluton layer based on FFT."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, padding: int = 0, stride: int = 1,
                 bias: bool = True, padding_mode: str = "zeros"):
        super().__init__(in_channels, out_channels, kernel_size, padding=padding, stride=stride, bias=bias, padding_mode=padding_mode)
        self.bias = nn.Parameter(torch.randn(out_channels, )) if bias else None
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))


class BlurOp(nn.Module):
    def __init__(self,inp_size,init_kernel=None,kernel_size=701,rd_params=None,rd_fac=1e-2):
        super(BlurOp, self).__init__()
        assert kernel_size % 2 == 1     #  kernel size must be uneven in order to simplify the passing of the padding argument

        self.inp_size = inp_size
        self.shape = inp_size
        self.kernel_size = kernel_size
        
        self.conv2d = FFTConv2d(in_channels=1,
                                out_channels=1,
                                kernel_size=kernel_size,
                                padding=kernel_size // 2,
                                padding_mode='replicate',
                                bias=False)

        if init_kernel is not None:
            self.conv2d.weight.data = init_kernel.expand(self.conv2d.weight.shape).clone()
        else:
            torch.nn.init.zeros_(self.conv2d.weight)

        if rd_params is None:
            rd_params = torch.zeros(2, requires_grad=False)

        self.rd_fac = torch.nn.Parameter(torch.tensor(rd_fac), requires_grad=False)
        self.rd_params = torch.nn.Parameter(rd_fac * rd_params, requires_grad=rd_params.requires_grad)

        dom_x = torch.linspace(-1.0, 1.0, inp_size[1])
        dom_y = torch.linspace(-1.0, 1.0, inp_size[1])
        self.grid_x, self.grid_y = torch.meshgrid(dom_x, dom_y)
        self.grid_x = torch.nn.Parameter(self.grid_x.unsqueeze(0).clone(), requires_grad=False) # register_buffer
        self.grid_y = torch.nn.Parameter(self.grid_y.unsqueeze(0).clone(), requires_grad=False)
        self.rad = torch.nn.Parameter((self.grid_x ** 2 + self.grid_y ** 2).sqrt(), requires_grad=False)


    def undistort(self, x):
        rad_fac_x = 1 + self.rd_params[0] / self.rd_fac * self.rad ** 2 + \
                    self.rd_params[1] / self.rd_fac * self.rad ** 4
        rad_fac_y = 1 + self.rd_params[0] / self.rd_fac * self.rad ** 2 + \
                    self.rd_params[1] / self.rd_fac * self.rad ** 4
        rad_grid_x = self.grid_x / rad_fac_x
        rad_grid_y = self.grid_y / rad_fac_y
        rad_grid = torch.cat([rad_grid_x.unsqueeze(-1), rad_grid_y.unsqueeze(-1)], -1)
        rad_grid = torch.repeat_interleave(rad_grid, x.shape[0], dim=0)
        size_diff = x.shape[-1] - x.shape[-2]
        assert size_diff >= 0
        out_padded = torch.nn.functional.pad(x, (0, 0, size_diff // 2, size_diff // 2), "replicate")
        out_distorted = torch.nn.functional.grid_sample(out_padded, rad_grid.transpose(1, 2), padding_mode="border", align_corners=False)
        out_distorted = out_distorted[..., size_diff // 2:(x.shape[-1] - size_diff // 2), :]

        return out_distorted

    def distort(self, x):
        rad_fac_x = 1 + self.rd_params[0] / self.rd_fac * self.rad ** 2 + \
                    self.rd_params[1] / self.rd_fac * self.rad ** 4
        rad_fac_y = 1 + self.rd_params[0] / self.rd_fac * self.rad ** 2 + \
                    self.rd_params[1] / self.rd_fac * self.rad ** 4
        rad_grid_x = self.grid_x * rad_fac_x  # .expand(x.shape)
        rad_grid_y = self.grid_y * rad_fac_y
        rad_grid = torch.cat([rad_grid_x.unsqueeze(-1), rad_grid_y.unsqueeze(-1)], -1)
        rad_grid = torch.repeat_interleave(rad_grid, x.shape[0], dim=0)
        size_diff = x.shape[-1] - x.shape[-2]
        assert size_diff >= 0
        out_padded = torch.nn.functional.pad(x, (0, 0, size_diff // 2, size_diff // 2), "replicate")
        out_distorted = torch.nn.functional.grid_sample(out_padded, rad_grid.transpose(1, 2), padding_mode="border", align_corners=False)
        out_distorted = out_distorted[..., size_diff // 2:(x.shape[-1] - size_diff // 2), :]
        return out_distorted

    def forward(self, x):
        out = self.conv2d(x)
        return self.distort(out)

    def visualize_distortion(self):

        with torch.no_grad():
            # create test image with a grid 
            image = torch.zeros(1,1,1460, 2360)
            image[0,0,np.where((np.arange(1460) + 60) % 105 <= 12)[0], :] = 1.0
            image[0,0,:,np.where((np.arange(2360) + 20)% 160 <= 12)[0]] = 1.0
            image = image.to(self.rd_params.device)
            image_distort = self.distort(image)

        return image, image_distort

    def transpose_conv(self, x):

        return fft_conv_nd(x,torch.flip(self.conv2d.weight, [-1, -2]),  padding=self.kernel_size // 2, padding_mode='replicate')

    def wiener_filter(self, y):
        xhat = wiener_filter_torch(y, self.conv2d.weight)
        return xhat
        