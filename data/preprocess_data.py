'''
This file downsamples all the data to 128x128
'''
import SimpleITK as sitk
from PIL import Image
import numpy as np
import os


def resizeImg(img, out_size):
    if img.endswith(".png"):
        # Get original image dimensions
        original_image = sitk.ReadImage(img, imageIO="PNGImageIO")
    elif img.endswith(".tif"):
        original_image = sitk.ReadImage(img, imageIO="TIFFImageIO")
    orig_size = original_image.GetSize()
    orig_spacing = original_image.GetSpacing()
    origin = original_image.GetOrigin()
    direction = original_image.GetDirection()

    # Calculate output spacing
    out_spacing = [
        int(np.round(orig_spacing[0] * (orig_size[0] / out_size[0]))),
        int(np.round(orig_spacing[1] * (orig_size[1] / out_size[1])))
        # might have to put RGB channel here?
    ]

    # Set output dimensions
    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputOrigin(origin)
    resampler.SetOutputDirection(direction)
    resampler.SetSize(out_size)
    resampler.SetOutputSpacing(out_spacing)

    return resampler.Execute(original_image)


# Pre-process training dataset
image_num = 1

for img in os.listdir("./training/av"):
    if img.endswith(".png") or img.endswith(".tif"):
        img = "./training/av/" + img
        # png_img = Image.open(os.path.join("./training/av", img))
        new_img = resizeImg(img, (128, 128))
        path = "./training/av/processed_labels/{}_label_downsampled.png".format(image_num)
        sitk.WriteImage(new_img, path)
        image_num += 1

image_num = 1

for img in os.listdir("./training/images"):
    if img.endswith(".png") or img.endswith(".tif"):
        img = "./training/images/" + img
        new_img = resizeImg(img, (128, 128))
        path = "./training/images/processed_labels/{}_img_downsampled.tif".format(image_num)
        sitk.WriteImage(new_img, path)
        image_num += 1

# Pre-process test dataset
image_num = 1

for img in os.listdir("./testing/av"):
    if img.endswith(".png") or img.endswith(".tif"):
        img = "./testing/av/" + img
        # png_img = Image.open(os.path.join("./training/av", img))
        new_img = resizeImg(img, (128, 128))
        path = "./testing/av/processed_labels/{}_label_downsampled.png".format(image_num)
        sitk.WriteImage(new_img, path)
        image_num += 1

image_num = 1

for img in os.listdir("./testing/images"):
    if img.endswith(".png") or img.endswith(".tif"):
        img = "./testing/images/" + img
        new_img = resizeImg(img, (128, 128))
        path = "./testing/images/processed_labels/{}_img_downsampled.tif".format(image_num)
        sitk.WriteImage(new_img, path)
        image_num += 1