import matplotlib.image as mpimg
import numpy as np
import cv2
from skimage.feature import hog
from lesson_functions import get_hog_features
from lesson_functions import get_hog_features
# Define a function to compute binned color features
def visialize_bin_spatial(img, size=(32, 32)):
    # Use cv2.resize().ravel() to create the feature vector
    features = cv2.resize(img, size)
    # Return the feature vector
    return features

def visialize_hog_image_and_color_hist(image, color_space, orient, pix_per_cell, cell_per_block, nbins, bins_range, spatial_size):
    if color_space != 'RGB':
        if color_space == 'HSV':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        elif color_space == 'LUV':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
        elif color_space == 'HLS':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
        elif color_space == 'YUV':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
        elif color_space == 'YCrCb':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
    else: feature_image = np.copy(image)
    hog_images = []
    _hog_features = []
    channel_hist  = []
    for index in range(feature_image.shape[2]):
        ch = feature_image[:, :, index]
        features, hog_image = get_hog_features(ch, orient, pix_per_cell, cell_per_block, True, True)
        hog_images.append(hog_image)
        _hog_features.append(features)

        channel_hist.append(np.histogram(ch, bins=nbins, range=bins_range))

    hist_features = np.concatenate((channel_hist[0][0], channel_hist[1][0], channel_hist[2][0]))
    hog_features = np.ravel(_hog_features)

    spatial_image = visialize_bin_spatial(feature_image, size=spatial_size)

    return feature_image, hog_images, spatial_image, hist_features, hog_features
