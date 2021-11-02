# Copyright 2021 solo-learn development team.

# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to use,
# copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the
# Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies
# or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
# PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
# FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import os
from pathlib import Path
from typing import Callable, Iterable, List, Sequence, Union

import nvidia.dali.fn as fn
import nvidia.dali.ops as ops
import nvidia.dali.types as types
from nvidia.dali.pipeline import Pipeline


class Mux:
    def __init__(self, prob: float):
        """Implements mutex operation for dali in order to support probabilitic augmentations.

        Args:
            prob (float): probability value
        """

        self.to_bool = ops.Cast(dtype=types.DALIDataType.BOOL)
        self.rng = ops.random.CoinFlip(probability=prob)

    def __call__(self, true_case, false_case):
        condition = self.to_bool(self.rng())
        neg_condition = condition ^ True
        return condition * true_case + neg_condition * false_case


class RandomGrayScaleConversion:
    def __init__(self, prob: float = 0.2, device: str = "gpu"):
        """Converts image to greyscale with probability.

        Args:
            prob (float, optional): probability of conversion. Defaults to 0.2.
            device (str, optional): device on which the operation will be performed.
                Defaults to "gpu".
        """

        self.mux = Mux(prob=prob)
        self.grayscale = ops.ColorSpaceConversion(
            device=device, image_type=types.RGB, output_type=types.GRAY
        )

    def __call__(self, images):
        out = self.grayscale(images)
        out = fn.cat(out, out, out, axis=2)
        return self.mux(true_case=out, false_case=images)


class RandomColorJitter:
    def __init__(
        self,
        brightness: float,
        contrast: float,
        saturation: float,
        hue: float,
        prob: float = 0.8,
        device: str = "gpu",
    ):
        """Applies random color jittering with probability.

        Args:
            brightness (float): brightness value for samplying uniformly
                in [max(0, 1 - brightness), 1 + brightness].
            contrast (float): contrast value for samplying uniformly
                in [max(0, 1 - contrast), 1 + contrast].
            saturation (float): saturation value for samplying uniformly
                in [max(0, 1 - saturation), 1 + saturation].
            hue (float): hue value for samplying uniformly in [-hue, hue].
            prob (float, optional): probability of applying jitter. Defaults to 0.8.
            device (str, optional): device on which the operation will be performed.
                Defaults to "gpu".
        """

        assert 0 <= hue <= 0.5

        self.mux = Mux(prob=prob)

        self.color = ops.ColorTwist(device=device)

        # look at torchvision docs to see how colorjitter samples stuff
        # for bright, cont and sat, it samples from [1-v, 1+v]
        # for hue, it samples from [-hue, hue]

        self.brightness = 1
        self.contrast = 1
        self.saturation = 1
        self.hue = 0

        if brightness:
            self.brightness = ops.random.Uniform(range=[max(0, 1 - brightness), 1 + brightness])

        if contrast:
            self.contrast = ops.random.Uniform(range=[max(0, 1 - contrast), 1 + contrast])

        if saturation:
            self.saturation = ops.random.Uniform(range=[max(0, 1 - saturation), 1 + saturation])

        if hue:
            # dali uses hue in degrees for some reason...
            hue = 360 * hue
            self.hue = ops.random.Uniform(range=[-hue, hue])

    def __call__(self, images):
        out = self.color(
            images,
            brightness=self.brightness() if callable(self.brightness) else self.brightness,
            contrast=self.contrast() if callable(self.contrast) else self.contrast,
            saturation=self.saturation() if callable(self.saturation) else self.saturation,
            hue=self.hue() if callable(self.hue) else self.hue,
        )
        return self.mux(true_case=out, false_case=images)


class RandomGaussianBlur:
    def __init__(self, prob: float = 0.5, window_size: int = 23, device: str = "gpu"):
        """Applies random gaussian blur with probability.

        Args:
            prob (float, optional): probability of applying random gaussian blur. Defaults to 0.5.
            window_size (int, optional): window size for gaussian blur. Defaults to 23.
            device (str, optional): device on which the operation will be performe.
                Defaults to "gpu".
        """

        self.mux = Mux(prob=prob)
        # gaussian blur
        self.gaussian_blur = ops.GaussianBlur(device=device, window_size=(window_size, window_size))
        self.sigma = ops.random.Uniform(range=[0, 1])

    def __call__(self, images):
        sigma = self.sigma() * 1.9 + 0.1
        out = self.gaussian_blur(images, sigma=sigma)
        return self.mux(true_case=out, false_case=images)


class RandomSolarize:
    def __init__(self, threshold: int = 128, prob: float = 0.0):
        """Applies random solarization with probability.

        Args:
            threshold (int, optional): threshold for inversion. Defaults to 128.
            prob (float, optional): probability of solarization. Defaults to 0.0.
        """

        self.mux = Mux(prob=prob)

        self.threshold = threshold

    def __call__(self, images):
        inverted_img = 255 - images
        mask = images >= self.threshold
        out = mask * inverted_img + (True ^ mask) * images
        return self.mux(true_case=out, false_case=images)

class RandomHorizontalFlip:
    def __init__(self, prob: float = 0.5, device: str = "gpu"):
        """Applies random horizontal flip with probability.

        Args:
            prob (float, optional): probability of applying random horizontal flip. Defaults to 0.5.
            device (str, optional): device on which the operation will be performe.
                Defaults to "gpu".
        """

        self.mux = Mux(prob=prob)
        # gaussian blur
        self.horizontal_flip = ops.Flip(device=device, horizontal=1, vertical=0)

    def __call__(self, images):
        out = self.horizontal_flip(images)
        return self.mux(true_case=out, false_case=images)

class RandomVerticalFlip:
    def __init__(self, prob: float = 0.5, device: str = "gpu"):
        """Applies random vertical flip with probability.

        Args:
            prob (float, optional): probability of applying random vertical flip. Defaults to 0.5.
            device (str, optional): device on which the operation will be performe.
                Defaults to "gpu".
        """

        self.mux = Mux(prob=prob)
        # gaussian blur
        self.vertical_flip = ops.Flip(device=device, horizontal=0, vertical=1)

    def __call__(self, images):
        out = self.vertical_flip(images)
        return self.mux(true_case=out, false_case=images)


class NormalPipeline(Pipeline):
    def __init__(
        self,
        data_path: str,
        batch_size: int,
        device: str,
        validation: bool = False,
        device_id: int = 0,
        shard_id: int = 0,
        num_shards: int = 1,
        num_threads: int = 4,
        seed: int = 12,
    ):
        """Initializes the pipeline for validation or linear eval training.

        If validation is set to True then images will only be resized to 256px and center cropped
        to 224px, otherwise random resized crop, horizontal flip are applied. In both cases images
        are normalized.

        Args:
            data_path (str): directory that contains the data.
            batch_size (int): batch size.
            device (str): device on which the operation will be performed.
            validation (bool): whether it is validation or training. Defaults to False. Defaults to
                False.
            device_id (int): id of the device used to initialize the seed and for parent class.
                Defaults to 0.
            shard_id (int): id of the shard (chuck of samples). Defaults to 0.
            num_shards (int): total number of shards. Defaults to 1.
            num_threads (int): number of threads to run in parallel. Defaults to 4.
            seed (int): seed for random number generation. Defaults to 12.
        """

        seed += device_id
        super().__init__(batch_size, num_threads, device_id, seed)

        self.device = device
        self.validation = validation

        self.reader = ops.readers.File(
            file_root=data_path,
            shard_id=shard_id,
            num_shards=num_shards,
            shuffle_after_epoch=True if not self.validation else False,
        )
        decoder_device = "mixed" if self.device == "gpu" else "cpu"
        device_memory_padding = 211025920 if decoder_device == "mixed" else 0
        host_memory_padding = 140544512 if decoder_device == "mixed" else 0
        self.decode = ops.decoders.Image(
            device=decoder_device,
            output_type=types.RGB,
            device_memory_padding=device_memory_padding,
            host_memory_padding=host_memory_padding,
        )

        # crop operations
        if self.validation:
            self.resize = ops.Resize(
                device=self.device,
                resize_shorter=256,
                interp_type=types.INTERP_CUBIC,
            )
            # center crop and normalize
            self.cmn = ops.CropMirrorNormalize(
                device=self.device,
                dtype=types.FLOAT,
                output_layout=types.NCHW,
                crop=(224, 224),
                mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
                std=[0.228 * 255, 0.224 * 255, 0.225 * 255],
            )
        else:
            self.resize = ops.RandomResizedCrop(
                device=self.device,
                size=224,
                random_area=(0.08, 1.0),
                interp_type=types.INTERP_CUBIC,
            )
            # normalize and horizontal flip
            self.cmn = ops.CropMirrorNormalize(
                device=self.device,
                dtype=types.FLOAT,
                output_layout=types.NCHW,
                mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
                std=[0.228 * 255, 0.224 * 255, 0.225 * 255],
            )

        self.coin05 = ops.random.CoinFlip(probability=0.5)
        self.to_int64 = ops.Cast(dtype=types.INT64, device=device)

    def define_graph(self):
        """Defines the computational graph for dali operations."""

        # read images from memory
        inputs, labels = self.reader(name="Reader")
        images = self.decode(inputs)

        # crop into large and small images
        images = self.resize(images)

        if self.validation:
            # crop and normalize
            images = self.cmn(images)
        else:
            # normalize and maybe apply horizontal flip with 0.5 chance
            images = self.cmn(images, mirror=self.coin05())

        if self.device == "gpu":
            labels = labels.gpu()
        # PyTorch expects labels as INT64
        labels = self.to_int64(labels)

        return (images, labels)


class CustomNormalPipeline(NormalPipeline):
    """Initializes the custom pipeline for validation or linear eval training.
    This acts as a placeholder and behaves exactly like NormalPipeline.
    If you want to do exoteric augmentations, you can just re-write this class.
    """

    pass


class ImagenetTransform:
    def __init__(
        self,
        device: str,
        brightness: float,
        contrast: float,
        saturation: float,
        hue: float,
        gaussian_prob: float = 0.5,
        solarization_prob: float = 0.0,
        size: int = 224,
        min_scale: float = 0.08,
        max_scale: float = 1.0,
    ):
        """Applies Imagenet transformations to a batch of images.

        Args:
            device (str): device on which the operations will be performed.
            brightness (float): sampled uniformly in [max(0, 1 - brightness), 1 + brightness].
            contrast (float): sampled uniformly in [max(0, 1 - contrast), 1 + contrast].
            saturation (float): sampled uniformly in [max(0, 1 - saturation), 1 + saturation].
            hue (float): sampled uniformly in [-hue, hue].
            gaussian_prob (float, optional): probability of applying gaussian blur. Defaults to 0.5.
            solarization_prob (float, optional): probability of applying solarization. Defaults
                to 0.0.
            size (int, optional): size of the side of the image after transformation. Defaults
                to 224.
            min_scale (float, optional): minimum scale of the crops. Defaults to 0.08.
            max_scale (float, optional): maximum scale of the crops. Defaults to 1.0.
        """

        # random crop
        self.random_crop = ops.RandomResizedCrop(
            device=device,
            size=size,
            random_area=(min_scale, max_scale),
            interp_type=types.INTERP_CUBIC,
        )

        # color jitter
        self.random_color_jitter = RandomColorJitter(
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue,
            prob=0.8,
            device=device,
        )

        # grayscale conversion
        self.random_grayscale = RandomGrayScaleConversion(prob=0.2, device=device)

        # gaussian blur
        self.random_gaussian_blur = RandomGaussianBlur(prob=gaussian_prob, device=device)

        # solarization
        self.random_solarization = RandomSolarize(prob=solarization_prob)

        # normalize and horizontal flip
        self.cmn = ops.CropMirrorNormalize(
            device=device,
            dtype=types.FLOAT,
            output_layout=types.NCHW,
            mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
            std=[0.228 * 255, 0.224 * 255, 0.225 * 255],
        )
        self.coin05 = ops.random.CoinFlip(probability=0.5)

        self.str = (
            "ImagenetTransform("
            f"random_crop({min_scale}, {max_scale}), "
            f"random_color_jitter(brightness={brightness}, "
            f"contrast={contrast}, saturation={saturation}, hue={hue}), "
            f"random_gray_scale, random_gaussian_blur({gaussian_prob}), "
            f"random_solarization({solarization_prob}), "
            "crop_mirror_resize())"
        )

    def __str__(self) -> str:
        return self.str

    def __call__(self, images):
        out = self.random_crop(images)
        out = self.random_color_jitter(out)
        out = self.random_grayscale(out)
        out = self.random_gaussian_blur(out)
        out = self.random_solarization(out)
        out = self.cmn(out, mirror=self.coin05())
        return out


class ModifiedImagenetTransform_4Branch:
    def __init__(
        self,
        device: str,
        brightness: float,
        contrast: float,
        saturation: float,
        hue: float,
        gaussian_prob: float = 0.5,
        solarization_prob: float = 0.0,
        size: int = 224,
        min_scale: float = 0.08,
        max_scale: float = 1.0,
    ):
        """Applies Imagenet transformations to a batch of images.

        Args:
            device (str): device on which the operations will be performed.
            brightness (float): sampled uniformly in [max(0, 1 - brightness), 1 + brightness].
            contrast (float): sampled uniformly in [max(0, 1 - contrast), 1 + contrast].
            saturation (float): sampled uniformly in [max(0, 1 - saturation), 1 + saturation].
            hue (float): sampled uniformly in [-hue, hue].
            gaussian_prob (float, optional): probability of applying gaussian blur. Defaults to 0.5.
            solarization_prob (float, optional): probability of applying solarization. Defaults
                to 0.0.
            size (int, optional): size of the side of the image after transformation. Defaults
                to 224.
            min_scale (float, optional): minimum scale of the crops. Defaults to 0.08.
            max_scale (float, optional): maximum scale of the crops. Defaults to 1.0.
        """
        # random horizontal flip
        self.random_horizontal_flip = RandomHorizontalFlip(prob=0.5, device=device)

        # random vertical flip
        self.random_vertical_flip = RandomVerticalFlip(prob=0.5, device=device)

        # random crop
        self.random_crop = ops.RandomResizedCrop(
            device=device,
            size=size,
            random_area=(min_scale, max_scale),
            interp_type=types.INTERP_CUBIC,
        )

        # random crop
        self.crp_random_crop = ops.RandomResizedCrop(
            device=device,
            size=size,
            random_area=(0.5, max_scale),
            interp_type=types.INTERP_CUBIC,
        )

        # color jitter
        self.random_color_jitter = RandomColorJitter(
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue,
            prob=0.8,
            device=device,
        )

        # grayscale conversion
        self.random_grayscale = RandomGrayScaleConversion(prob=0.2, device=device)

        # gaussian blur
        self.random_gaussian_blur = RandomGaussianBlur(prob=gaussian_prob, device=device)

        # solarization
        self.random_solarization = RandomSolarize(prob=solarization_prob)

        # normalize
        self.cmn = ops.CropMirrorNormalize(
            device=device,
            dtype=types.FLOAT,
            output_layout=types.NCHW,
            mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
            std=[0.228 * 255, 0.224 * 255, 0.225 * 255],
        )
        self.coin05 = ops.random.CoinFlip(probability=0)

        self.str = (
            "ImagenetTransform("
            f"random_horizontal_flip(0.5), "
            f"random_crop({min_scale}, {max_scale}), "
            f"random_color_jitter(brightness={brightness}, "
            f"contrast={contrast}, saturation={saturation}, hue={hue}), "
            f"random_gray_scale, random_gaussian_blur({gaussian_prob}), "
            f"random_solarization({solarization_prob}), "
            "crop_mirror_resize())"
        )

    def __str__(self) -> str:
        return self.str

    # 这段代码batch中的2,3 生成的是对齐的crop
    # def __call__(self, images):
    #     out0 = self.random_crop(images)
    #     out0 = self.random_horizontal_flip(out0)
    #     out1 = self.random_crop(images)
    #     out1 = self.random_horizontal_flip(out1)
    #     out23 = self.random_horizontal_flip(images)
    #     out23 = self.crp_random_crop(out23)

        
    #     x0 = self.random_color_jitter(out0)
    #     x0 = self.random_grayscale(x0)
    #     x0 = self.random_gaussian_blur(x0)
    #     x0 = self.random_solarization(x0)
    #     x0 = self.cmn(x0, mirror=self.coin05())

    #     x1 = self.random_color_jitter(out1)
    #     x1 = self.random_grayscale(x1)
    #     x1 = self.random_gaussian_blur(x1)
    #     x1 = self.random_solarization(x1)
    #     x1 = self.cmn(x1, mirror=self.coin05())

    #     x2 = self.random_color_jitter(out23)
    #     x2 = self.random_grayscale(x2)
    #     x2 = self.random_gaussian_blur(x2)
    #     x2 = self.random_solarization(x2)
    #     x2 = self.cmn(x2, mirror=self.coin05())

    #     x3 = self.random_color_jitter(out23)
    #     x3 = self.random_grayscale(x3)
    #     x3 = self.random_gaussian_blur(x3)
    #     x3 = self.random_solarization(x3)
    #     x3 = self.cmn(x3, mirror=self.coin05())

    #     return [x0, x1, x2, x3]

    # 这段代码batch中的2,3 生成的是非对齐的crop，但享有相同的horizontal filp
    def __call__(self, images):
        out0 = self.random_crop(images)
        out0 = self.random_horizontal_flip(out0)
        out1 = self.random_crop(images)
        out1 = self.random_horizontal_flip(out1)
        out23 = self.random_horizontal_flip(images)
        out2 = self.random_crop(out23)
        out3 = self.random_crop(out23)

        
        x0 = self.random_color_jitter(out0)
        x0 = self.random_grayscale(x0)
        x0 = self.random_gaussian_blur(x0)
        x0 = self.random_solarization(x0)
        x0 = self.cmn(x0, mirror=self.coin05())

        x1 = self.random_color_jitter(out1)
        x1 = self.random_grayscale(x1)
        x1 = self.random_gaussian_blur(x1)
        x1 = self.random_solarization(x1)
        x1 = self.cmn(x1, mirror=self.coin05())

        x2 = self.random_color_jitter(out2)
        x2 = self.random_grayscale(x2)
        x2 = self.random_gaussian_blur(x2)
        x2 = self.random_solarization(x2)
        x2 = self.cmn(x2, mirror=self.coin05())

        x3 = self.random_color_jitter(out3)
        x3 = self.random_grayscale(x3)
        x3 = self.random_gaussian_blur(x3)
        x3 = self.random_solarization(x3)
        x3 = self.cmn(x3, mirror=self.coin05())

        return [x0, x1, x2, x3]

class ModifiedImagenetTransform_6Branch:
    def __init__(
        self,
        device: str,
        brightness: float,
        contrast: float,
        saturation: float,
        hue: float,
        gaussian_prob: float = 0.5,
        solarization_prob: float = 0.0,
        size: int = 224,
        min_scale: float = 0.08,
        max_scale: float = 1.0,
    ):
        """Applies Imagenet transformations to a batch of images.

        Args:
            device (str): device on which the operations will be performed.
            brightness (float): sampled uniformly in [max(0, 1 - brightness), 1 + brightness].
            contrast (float): sampled uniformly in [max(0, 1 - contrast), 1 + contrast].
            saturation (float): sampled uniformly in [max(0, 1 - saturation), 1 + saturation].
            hue (float): sampled uniformly in [-hue, hue].
            gaussian_prob (float, optional): probability of applying gaussian blur. Defaults to 0.5.
            solarization_prob (float, optional): probability of applying solarization. Defaults
                to 0.0.
            size (int, optional): size of the side of the image after transformation. Defaults
                to 224.
            min_scale (float, optional): minimum scale of the crops. Defaults to 0.08.
            max_scale (float, optional): maximum scale of the crops. Defaults to 1.0.
        """
        # random horizontal flip
        self.random_horizontal_flip = RandomHorizontalFlip(prob=0.5, device=device)

        # random vertical flip
        self.random_vertical_flip = RandomVerticalFlip(prob=0.5, device=device)

        # random crop
        self.random_crop = ops.RandomResizedCrop(
            device=device,
            size=size,
            random_area=(min_scale, max_scale),
            interp_type=types.INTERP_CUBIC,
        )

        # random crop
        self.crp_random_crop = ops.RandomResizedCrop(
            device=device,
            size=size,
            random_area=(0.25, max_scale),
            interp_type=types.INTERP_CUBIC,
        )

        # color jitter
        self.random_color_jitter = RandomColorJitter(
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue,
            prob=0.8,
            device=device,
        )

        # grayscale conversion
        self.random_grayscale = RandomGrayScaleConversion(prob=0.2, device=device)

        # gaussian blur
        self.random_gaussian_blur = RandomGaussianBlur(prob=gaussian_prob, device=device)

        # solarization
        self.random_solarization = RandomSolarize(prob=solarization_prob)

        # normalize
        self.cmn = ops.CropMirrorNormalize(
            device=device,
            dtype=types.FLOAT,
            output_layout=types.NCHW,
            mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
            std=[0.228 * 255, 0.224 * 255, 0.225 * 255],
        )
        self.coin05 = ops.random.CoinFlip(probability=0)

        self.str = (
            "ImagenetTransform("
            f"random_horizontal_flip(0.5), "
            f"random_crop({min_scale}, {max_scale}), "
            f"random_color_jitter(brightness={brightness}, "
            f"contrast={contrast}, saturation={saturation}, hue={hue}), "
            f"random_gray_scale, random_gaussian_blur({gaussian_prob}), "
            f"random_solarization({solarization_prob}), "
            "crop_mirror_resize())"
        )

    def __str__(self) -> str:
        return self.str

    def __call__(self, images):
        out0 = self.random_crop(images)
        out0 = self.random_horizontal_flip(out0)
        out1 = self.random_crop(images)
        out1 = self.random_horizontal_flip(out1)
        out23 = self.random_horizontal_flip(images)
        out23 = self.crp_random_crop(out23)
        out4 = self.random_crop(images)
        out4 = self.random_horizontal_flip(out4)
        out5 = self.random_crop(images)
        out5 = self.random_horizontal_flip(out5)
        
        x0 = self.random_color_jitter(out0)
        x0 = self.random_grayscale(x0)
        x0 = self.random_gaussian_blur(x0)
        x0 = self.random_solarization(x0)
        x0 = self.cmn(x0, mirror=self.coin05())

        x1 = self.random_color_jitter(out1)
        x1 = self.random_grayscale(x1)
        x1 = self.random_gaussian_blur(x1)
        x1 = self.random_solarization(x1)
        x1 = self.cmn(x1, mirror=self.coin05())

        x2 = self.random_color_jitter(out23)
        x2 = self.random_grayscale(x2)
        x2 = self.random_gaussian_blur(x2)
        x2 = self.random_solarization(x2)
        x2 = self.cmn(x2, mirror=self.coin05())

        x3 = self.random_color_jitter(out23)
        x3 = self.random_grayscale(x3)
        x3 = self.random_gaussian_blur(x3)
        x3 = self.random_solarization(x3)
        x3 = self.cmn(x3, mirror=self.coin05())

        x4= self.random_color_jitter(out4)
        x4 = self.random_grayscale(x4)
        x4 = self.random_gaussian_blur(x4)
        x4 = self.random_solarization(x4)
        x4 = self.cmn(x4, mirror=self.coin05())

        x5= self.random_color_jitter(out5)
        x5 = self.random_grayscale(x5)
        x5 = self.random_gaussian_blur(x5)
        x5 = self.random_solarization(x5)
        x5 = self.cmn(x5, mirror=self.coin05())

        return [x0, x1, x2, x3, x4, x5]

class CustomTransform:
    def __init__(
        self,
        device: str,
        brightness: float,
        contrast: float,
        saturation: float,
        hue: float,
        gaussian_prob: float = 0.5,
        solarization_prob: float = 0.0,
        size: int = 224,
        min_scale: float = 0.08,
        max_scale: float = 1.0,
        mean: Sequence[float] = (0.485, 0.456, 0.406),
        std: Sequence[float] = (0.228, 0.224, 0.225),
    ):
        """Applies Custom transformations.
        If you want to do exoteric augmentations, you can just re-write this class.

        Args:
            device (str): device on which the operations will be performed.
            brightness (float): sampled uniformly in [max(0, 1 - brightness), 1 + brightness].
            contrast (float): sampled uniformly in [max(0, 1 - contrast), 1 + contrast].
            saturation (float): sampled uniformly in [max(0, 1 - saturation), 1 + saturation].
            hue (float): sampled uniformly in [-hue, hue].
            gaussian_prob (float, optional): probability of applying gaussian blur. Defaults to 0.5.
            solarization_prob (float, optional): probability of applying solarization. Defaults
                to 0.0.
            size (int, optional): size of the side of the image after transformation. Defaults
                to 224.
            min_scale (float, optional): minimum scale of the crops. Defaults to 0.08.
            max_scale (float, optional): maximum scale of the crops. Defaults to 1.0.
            mean (Sequence[float], optional): mean values for normalization.
                Defaults to (0.485, 0.456, 0.406).
            std (Sequence[float], optional): std values for normalization.
                Defaults to (0.228, 0.224, 0.225).
        """

        # random crop
        self.random_crop = ops.RandomResizedCrop(
            device=device,
            size=size,
            random_area=(min_scale, max_scale),
            interp_type=types.INTERP_CUBIC,
        )

        # color jitter
        self.random_color_jitter = RandomColorJitter(
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue,
            prob=0.8,
            device=device,
        )

        # grayscale conversion
        self.random_grayscale = RandomGrayScaleConversion(prob=0.2, device=device)

        # gaussian blur
        self.random_gaussian_blur = RandomGaussianBlur(prob=gaussian_prob, device=device)

        # solarization
        self.random_solarization = RandomSolarize(prob=solarization_prob)

        # normalize and horizontal flip
        self.cmn = ops.CropMirrorNormalize(
            device=device,
            dtype=types.FLOAT,
            output_layout=types.NCHW,
            mean=[v * 255 for v in mean],
            std=[v * 255 for v in std],
        )
        self.coin05 = ops.random.CoinFlip(probability=0.5)

        self.str = (
            "CustomTransform("
            f"random_crop({min_scale}, {max_scale}), "
            f"random_color_jitter(brightness={brightness}, "
            f"contrast={contrast}, saturation={saturation}, hue={hue}), "
            f"random_gray_scale, random_gaussian_blur({gaussian_prob}), "
            f"random_solarization({solarization_prob}), "
            "crop_mirror_resize())"
        )

    def __call__(self, images):
        out = self.random_crop(images)
        out = self.random_color_jitter(out)
        out = self.random_grayscale(out)
        out = self.random_gaussian_blur(out)
        out = self.random_solarization(out)
        out = self.cmn(out, mirror=self.coin05())
        return out

    def __str__(self):
        return self.str


class PretrainPipeline(Pipeline):
    def __init__(
        self,
        data_path: Union[str, Path],
        batch_size: int,
        device: str,
        transform: Union[Callable, Iterable],
        num_crops: int = 2,
        random_shuffle: bool = True,
        device_id: int = 0,
        shard_id: int = 0,
        num_shards: int = 1,
        num_threads: int = 4,
        seed: int = 12,
        no_labels: bool = False,
        encode_indexes_into_labels: bool = False,
    ):
        """Initializes the pipeline for pretraining.

        Args:
            data_path (str): directory that contains the data.
            batch_size (int): batch size.
            device (str): device on which the operation will be performed.
            transform (Union[Callable, Iterable]): a transformation or a sequence
                of transformations to be applied.
            num_crops (int, optional): number of crops. Defaults to 2.
            random_shuffle (bool, optional): whether to randomly shuffle the samples.
                Defaults to True.
            device_id (int, optional): id of the device used to initialize the seed and
                for parent class. Defaults to 0.
            shard_id (int, optional): id of the shard (chuck of samples). Defaults to 0.
            num_shards (int, optional): total number of shards. Defaults to 1.
            num_threads (int, optional): number of threads to run in parallel. Defaults to 4.
            seed (int, optional): seed for random number generation. Defaults to 12.
            no_labels (bool, optional): if the data has no labels. Defaults to False.
            encode_indexes_into_labels (bool, optional): uses sample indexes as labels
                and then gets the labels from a lookup table. This may use more CPU memory,
                so just use when needed. Defaults to False.
        """

        seed += device_id
        super().__init__(
            batch_size=batch_size,
            num_threads=num_threads,
            device_id=device_id,
            seed=seed,
        )

        self.device = device

        data_path = Path(data_path)
        if no_labels:
            files = [data_path / f for f in sorted(os.listdir(data_path))]
            labels = [-1] * len(files)
            self.reader = ops.readers.File(
                files=files,
                shard_id=shard_id,
                num_shards=num_shards,
                shuffle_after_epoch=random_shuffle,
                labels=labels,
            )
        elif encode_indexes_into_labels:
            labels = sorted(Path(entry.name) for entry in os.scandir(data_path) if entry.is_dir())

            data = [
                (data_path / label / file, label_idx)
                for label_idx, label in enumerate(labels)
                for file in sorted(os.listdir(data_path / label))
            ]

            files = []
            labels = []
            # for debugging
            true_labels = []

            self.conversion_map = []
            for file_idx, (file, label_idx) in enumerate(data):
                files.append(file)
                labels.append(file_idx)
                true_labels.append(label_idx)
                self.conversion_map.append(label_idx)

            # debugging
            for file, file_idx, label_idx in zip(files, labels, true_labels):
                assert self.conversion_map[file_idx] == label_idx

            self.reader = ops.readers.File(
                files=files,
                shard_id=shard_id,
                num_shards=num_shards,
                shuffle_after_epoch=random_shuffle,
            )
        else:
            self.reader = ops.readers.File(
                file_root=data_path,
                shard_id=shard_id,
                num_shards=num_shards,
                shuffle_after_epoch=random_shuffle,
            )

        decoder_device = "mixed" if self.device == "gpu" else "cpu"
        device_memory_padding = 211025920 if decoder_device == "mixed" else 0
        host_memory_padding = 140544512 if decoder_device == "mixed" else 0
        self.decode = ops.decoders.Image(
            device=decoder_device,
            output_type=types.RGB,
            device_memory_padding=device_memory_padding,
            host_memory_padding=host_memory_padding,
        )
        self.to_int64 = ops.Cast(dtype=types.INT64, device=device)

        self.num_crops = num_crops

        # transformations
        self.transform = transform

        if isinstance(transform, Iterable):
            self.one_transform_per_crop = True
        else:
            self.one_transform_per_crop = False
            self.num_crops = num_crops

    def define_graph(self):
        """Defines the computational graph for dali operations."""

        # read images from memory
        inputs, labels = self.reader(name="Reader")

        images = self.decode(inputs)

        if self.one_transform_per_crop:
            crops = [transform(images) for transform in self.transform]
        else:
            crops = [self.transform(images) for i in range(self.num_crops)]

        if self.device == "gpu":
            labels = labels.gpu()
        # PyTorch expects labels as INT64
        labels = self.to_int64(labels)

        return (*crops, labels)

class ModifiedPretrainPipeline(PretrainPipeline):
    def define_graph(self):
        """Defines the computational graph for dali operations."""

        # read images from memory
        inputs, labels = self.reader(name="Reader")

        images = self.decode(inputs)

        crops = self.transform(images)

        if self.device == "gpu":
            labels = labels.gpu()
        # PyTorch expects labels as INT64
        labels = self.to_int64(labels)

        return (*crops, labels)


class MulticropPretrainPipeline(Pipeline):
    def __init__(
        self,
        data_path: Union[str, Path],
        batch_size: int,
        device: str,
        transforms: List,
        num_crops: List[int],
        random_shuffle: bool = True,
        device_id: int = 0,
        shard_id: int = 0,
        num_shards: int = 1,
        num_threads: int = 4,
        seed: int = 12,
        no_labels: bool = False,
        encode_indexes_into_labels: bool = False,
    ):
        """Initializes the pipeline for pretraining with multicrop.

        Args:
            data_path (str): directory that contains the data.
            batch_size (int): batch size.
            device (str): device on which the operation will be performed.
            transforms (List): list of transformations to be applied.
            num_crops (List[int]): number of crops.
            random_shuffle (bool, optional): whether to randomly shuffle the samples.
                Defaults to True.
            device_id (int, optional): id of the device used to initialize the seed and
                for parent class. Defaults to 0.
            shard_id (int, optional): id of the shard (chuck of samples). Defaults to 0.
            num_shards (int, optional): total number of shards. Defaults to 1.
            num_threads (int, optional): number of threads to run in parallel. Defaults to 4.
            seed (int, optional): seed for random number generation. Defaults to 12.
            no_labels (bool, optional): if the data has no labels. Defaults to False.
            encode_indexes_into_labels (bool, optional): uses sample indexes as labels
                and then gets the labels from a lookup table. This may use more CPU memory,
                so just use when needed. Defaults to False.
        """

        seed += device_id
        super().__init__(
            batch_size=batch_size,
            num_threads=num_threads,
            device_id=device_id,
            seed=seed,
        )

        self.device = device

        data_path = Path(data_path)
        if no_labels:
            files = [data_path / f for f in sorted(os.listdir(data_path))]
            labels = [-1] * len(files)
            self.reader = ops.readers.File(
                files=files,
                shard_id=shard_id,
                num_shards=num_shards,
                shuffle_after_epoch=random_shuffle,
                labels=labels,
            )
        elif encode_indexes_into_labels:
            labels = sorted(Path(entry.name) for entry in os.scandir(data_path) if entry.is_dir())

            data = [
                (data_path / label / file, label_idx)
                for label_idx, label in enumerate(labels)
                for file in sorted(os.listdir(data_path / label))
            ]

            files = []
            labels = []
            # for debugging
            true_labels = []

            self.conversion_map = []
            for file_idx, (file, label_idx) in enumerate(data):
                files.append(file)
                labels.append(file_idx)
                true_labels.append(label_idx)
                self.conversion_map.append(label_idx)

            # debugging
            for file, file_idx, label_idx in zip(files, labels, true_labels):
                assert self.conversion_map[file_idx] == label_idx

            self.reader = ops.readers.File(
                files=files,
                shard_id=shard_id,
                num_shards=num_shards,
                shuffle_after_epoch=random_shuffle,
            )
        else:
            self.reader = ops.readers.File(
                file_root=data_path,
                shard_id=shard_id,
                num_shards=num_shards,
                shuffle_after_epoch=random_shuffle,
            )

        decoder_device = "mixed" if self.device == "gpu" else "cpu"
        device_memory_padding = 211025920 if decoder_device == "mixed" else 0
        host_memory_padding = 140544512 if decoder_device == "mixed" else 0
        self.decode = ops.decoders.Image(
            device=decoder_device,
            output_type=types.RGB,
            device_memory_padding=device_memory_padding,
            host_memory_padding=host_memory_padding,
        )
        self.to_int64 = ops.Cast(dtype=types.INT64, device=device)

        self.num_crops = num_crops
        self.transforms = transforms

        assert len(transforms) == len(num_crops)

    def define_graph(self):
        """Defines the computational graph for dali operations."""

        # read images from memory
        inputs, labels = self.reader(name="Reader")
        images = self.decode(inputs)

        # crop into large and small images
        crops = []
        for i, transform in enumerate(self.transforms):
            for _ in range(self.num_crops[i]):
                crop = transform(images)
                crops.append(crop)

        if self.device == "gpu":
            labels = labels.gpu()
        # PyTorch expects labels as INT64
        labels = self.to_int64(labels)

        return (*crops, labels)
