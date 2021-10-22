'''
This file downsamples all the data to 128x128
'''
import SimpleITK as sitk
import numpy as np

def resizeImg(img, out_size):
    ## Get original image dimensions
    original_image = sitk.ReadImage(img, sitk.sitkInt32)
    orig_size = original_image.GetSize()
    orig_spacing = original_image.GetSpacing()
    origin = original_image.GetOrigin()
    direction = original_image.GetDirection()

    ## Calculate output spacing
    out_spacing = [
        int(np.round(orig_spacing[0] * (orig_size[0] / out_size[0]))),
        int(np.round(orig_spacing[1] * (orig_size[1] / out_size[1])))
        ## might have to put RGB channel here?
    ]

    ## Set output dimensions
    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputOrigin(origin)
    resampler.SetOutputDirection(direction)
    resampler.SetSize(out_size)
    resampler.SetOutputSpacing(out_spacing)

    return resampler.execute(original_image)

# Pre-process training dataset
image_num = 1

for img in "/training/av":
    new_img = resizeImg(img, (128, 128))
    sitk.WriteImage(new_img, "/processed_labels/{}_label_downsampled".format(image_num))
    image_num += 1

image_num = 1

for img in "/training/images":
    new_img = resizeImg(img, (128, 128))
    sitk.WriteImage(new_img, "/processed_img/{}_img_downsampled".format(image_num))
    image_num += 1

# Pre-process test dataset
image_num = 1

for img in "/testing/av":
    new_img = resizeImg(img, (128, 128))
    sitk.WriteImage(new_img, "/processed_labels/{}_label_downsampled".format(image_num))
    image_num += 1

image_num = 1

for img in "/testing/images":
    new_img = resizeImg(img, (128, 128))
    sitk.WriteImage(new_img, "/processed_img/img/{}_img_downsampled".format(image_num))
    image_num += 1
