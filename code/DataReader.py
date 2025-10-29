import numpy as np

"""This script implements the functions for reading data.
"""

def load_data(filename):
    """Load a given txt file.

    Args:
        filename: A string.

    Returns:
        raw_data: An array of shape [n_samples, 256].
        labels : An array of shape [n_samples,].
        
    """
    data= np.load(filename)
    x= data['x']
    y= data['y']
    return x, y

def train_valid_split(raw_data, labels, split_index):
	"""Split the original training data into a new training dataset
	and a validation dataset.
	n_samples = n_train_samples + n_valid_samples

	Args:
		raw_data: An array of shape [n_samples, 256].
        labels : An array of shape [n_samples,].
		split_index: An integer.

	"""
	return raw_data[:split_index], raw_data[split_index:], labels[:split_index], labels[split_index:]


def prepare_X(raw_X):
    """Extract features from raw_X as required.

    Args:
        raw_X: An array of shape [n_samples, 256].

    Returns:
        X: An array of shape [n_samples, n_features].
    """
    raw_image = raw_X.reshape((-1, 16, 16))

    
    # Feature 1: Measure of Symmetry
    ### START YOUR CODE HERE ###
    # We'll compute horizontal symmetry: mean absolute difference
    # between the image and its left-right flipped version.
    # A lower value means more symmetric; we'll invert so larger -> more symmetric.
    flipped = np.flip(raw_image, axis=2)
    # mean absolute difference per image
    mad = np.mean(np.abs(raw_image - flipped), axis=(1, 2))
    # convert to a symmetry score in [0, 1] by normalizing by max possible (assuming pixel range 0-1 or 0-255)
    # To be robust, normalize by the max observed mad (avoid division by zero)
    max_mad = np.max(mad)
    if max_mad == 0:
        symmetry = np.ones_like(mad)
    else:
        symmetry = 1.0 - (mad / max_mad)

    ### END YOUR CODE HERE ###
    # Feature 2: Measure of Intensity
    ### YOUR CODE HERE ###
    # Use the mean pixel intensity per image (normalized to [0,1] by dividing by max observed)
    mean_intensity = np.mean(raw_image, axis=(1, 2))
    max_int = np.max(mean_intensity)
    if max_int == 0:
        intensity = np.zeros_like(mean_intensity)
    else:
        intensity = mean_intensity / max_int
    ### END YOUR CODE HERE ###

    # Feature 3: Bias Term. Always 1.
    ### YOUR CODE HERE 
    bias = np.ones((raw_image.shape[0],), dtype=float)

    ### END YOUR CODE HERE 

    # Stack features together in the following order.
    # [Feature 3, Feature 1, Feature 2]
    ### YOUR CODE HERE
    X = np.stack([bias, symmetry, intensity], axis=1)
    return X
    ### END YOUR CODE

def prepare_y(raw_y):
    """
    Args:
        raw_y: An array of shape [n_samples,].
        
    Returns:
        y: An array of shape [n_samples,].
        idx:return idx for data label 1 and 2.
    """
    y = raw_y
    idx = np.where((raw_y==1) | (raw_y==2))
    y[np.where(raw_y==0)] = 0
    y[np.where(raw_y==1)] = 1
    y[np.where(raw_y==2)] = 2

    return y, idx




