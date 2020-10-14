import os
import numpy as np
import cv2
from utils import extract_patches, pad
import openslide
from openslide_python_fix import _load_image_lessthan_2_29, _load_image_morethan_2_29

dir_images_panda = '../data/slides'
dirOut = '../data/patches'
patch_size = 512


def func_read_patch(slide, h, w):
    # Check which _load_image() function to use depending on the size of the region.
    if (h * w) >= 2**29:
        openslide.lowlevel._load_image = _load_image_morethan_2_29
    else:
        openslide.lowlevel._load_image = _load_image_lessthan_2_29
    region = slide.read_region((0,0), 0, (w, h)).convert('RGB')
    return region


images = os.listdir(dir_images_panda)

existing_images = os.listdir(dirOut)
existing_images = [i.split('_')[0] + str('.tiff') for i in existing_images]
existing_images = np.unique(np.array(existing_images))
existing_images = list(existing_images)

images = [i for i in images if i not in existing_images]

counter = 0
for iImage in images:

    print(str(counter+1) + ' / ' + str(len(images)))

    openslide_image = openslide.OpenSlide(os.path.join(dir_images_panda, iImage))
    downsamples = openslide_image.level_downsamples
    [w, h] = openslide_image.level_dimensions[0]
    size1 = int(w*(downsamples[0]/downsamples[2]))
    size2 = int(h*(downsamples[0]/downsamples[2]))

    # Computational limitations
    if size1*size2 < 4830920:

        slide = func_read_patch(openslide_image, h, w)

        openslide_image = None

        slide = np.array(slide)

        # Zero-padding if the slide size is smaller than patch-size
        if slide.shape[0] < patch_size*2 or slide.shape[1] < patch_size*2:
            slide = pad(slide, max(patch_size*2, slide.shape[0]), max(patch_size*2, slide.shape[1]))

            # Otsu's threshold for obtaining tissue mask
        gray = cv2.cvtColor(slide, cv2.COLOR_BGR2GRAY)
        ret2, th2 = cv2.threshold(gray, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        gray = None

        th2 = (np.reshape(th2, (th2.shape[0], th2.shape[1], 1)))

        # Extract patches form the slide
        patch = np.array(extract_patches(slide, window_shape=(patch_size*2, patch_size*2, 3)))
        slide = None
        patch_mask = np.array(extract_patches(th2, window_shape=(patch_size*2, patch_size*2, 1)))
        th2 = None

        # filter patches with less than 20% tissue and normalize images
        patches = []
        for i in range(patch.shape[0]):
            pTissue = 1 - np.sum(np.array(patch_mask[i, :, :, :])) / np.prod(np.array(patch_mask[i, :, :, :].shape))
            if pTissue > .20:
                batch_i = patch[i, :, :, :]
                batch_i = cv2.resize(batch_i, (patch_size, patch_size))
                patches.append(batch_i)

        patch = None
        patch_mask = None

        counter2 = 0
        for iPatch in patches:
            counter2 += 1
            iPatch = np.array(iPatch)
            cv2.imwrite(os.path.join(dirOut, iImage[:-5] + '_' + str(counter2) + str('.jpg')), iPatch)

        patches = None

    else:
        print('Image too large')
    counter += 1