#! /usr/bin/env python

"""Apply various mean and enhancement filters to images.

This program will allow the user to apply a mean or enhancement filter
to the test image of a bowl.  The user may select from one of several
filters, or supply their own 3x3 or 5x5 filter.

Author:
    Matthew Bourque, 10/16

Use:
    This program is intended to be executed from the command line as
    such:
        >>> python convolution_mask.py

    The user will then be prompted to either select a pre-defined
    filter, or supply their own.

Dependencies:
    The user must have a Python 2.7 installion.  The numpy and skimage
    external libraries are also required.
"""

import numpy as np
from skimage.io import imread
from skimage.io import imsave

# -----------------------------------------------------------------------------

def apply_mask(filter_mask):
    """Apply the given filter mask to a test image.

    Parameters
    ----------
    filter_mask : tuple
        A tuple whose 0th element is the mask array and whose 1st
        element is the name of the mask.
    """

    mask = filter_mask[0]
    mask_name = filter_mask[1]
    print 'Applying {}'.format(mask_name)

    # Test image
    image = imread('bowl.jpg')
    image_with_noise = imread('bowl_with_noise.jpg')

    # Initializations
    modified_image = np.copy(image).astype(float)
    modified_image_with_noise = np.copy(image_with_noise).astype(float)

    if mask.size == 9:
        start = 1
        nrows = image.shape[0] - 1
        ncols = image.shape[1] - 1
    elif mask.size == 25:
        start = 2
        nrows = image.shape[0] - 2
        ncols = image.shape[1] - 2

    # Iterate over the image
    for row in range(start,nrows):
        for col in range(start,ncols):

            # Gather the local neighborhood
            if mask.size == 9:
                neighborhood = get_neighborhood(image, row, col, '3x3')
                neighborhood_with_noise = get_neighborhood(image_with_noise, row, col, '3x3')
            elif mask.size == 25:
                neighborhood = get_neighborhood(image, row, col, '5x5')
                neighborhood_with_noise = get_neighborhood(image_with_noise, row, col, '5x5')

            # Convolve the neighborhood with the mask
            solution = np.sum(neighborhood * mask)
            solution_with_noise = np.sum(neighborhood_with_noise * mask)

            # Replace the central pixel with the convolution solution
            modified_image[row,col] = solution
            modified_image_with_noise[row,col] = solution_with_noise

    # Normalize the images to -1, 1 (required by skimage)
    modified_image /= np.max(np.abs(modified_image))
    modified_image_with_noise /= np.max(np.abs(modified_image_with_noise))

    # Write out new image
    imsave('bowl_with_{}.jpg'.format(mask_name), modified_image)
    print '\nFile written to bowl_with_{},jpg'.format(mask_name)
    imsave('bowl_with_{}_with_noise.jpg'.format(mask_name), modified_image_with_noise)
    print 'File written to bowl_with_{}_with_noise,jpg'.format(mask_name)

# -----------------------------------------------------------------------------

def convert_user_mask_to_array(mask_elements):
    """Create a 3x3 or 5x5 mask for the given list of individual
    mask elements.

    Parameters
    ----------
    mask_elements : list of floats
        A list of floats from which to construct the 3x3 or 5x5 mask.
        If there are 9 elements in the list, a 3x3 mask is constructed.
        If there are 25 elements, a 5x5 mask is constructed.

    Returns
    -------
    mask : 2D array
        A 3x3 or 5x5 2D array corresponding to the user-suppled mask
        elements.
    """

    if len(mask_elements) == 9:
        mask = np.array([[mask_elements[0], mask_elements[1], mask_elements[2]],
                         [mask_elements[3], mask_elements[4], mask_elements[5]],
                         [mask_elements[6], mask_elements[7], mask_elements[8]]])

    if len(mask_elements) == 25:
        mask = np.array([[mask_elements[0], mask_elements[1], mask_elements[2], mask_elements[3], mask_elements[4]],
                         [mask_elements[5], mask_elements[6], mask_elements[7], mask_elements[8], mask_elements[9]],
                         [mask_elements[10], mask_elements[11], mask_elements[12], mask_elements[13], mask_elements[14]],
                         [mask_elements[15], mask_elements[16], mask_elements[17], mask_elements[18], mask_elements[19]],
                         [mask_elements[20], mask_elements[21], mask_elements[22], mask_elements[23], mask_elements[24]]])

    return mask

# -----------------------------------------------------------------------------

def get_filter_dict():
    """Construct a dictionary holding the 6 pre-defined masks and their
    names.

    Returns
    -------
    filter_dict : dictionary
        A dictionary whose keys are integer values corresponding to the
        mask number, and whose values are tuples.  The 0th element of
        the tuple is the 2D mask array, and the 1st element is the name
        of the mask.
    """

    # Define the different types of masks
    mean_filter1 = np.array([[1/9., 1/9., 1/9.],
                             [1/9., 1/9., 1/9.],
                             [1/9., 1/9., 1/9.]])
    mean_filter2 = np.array([[1/10., 1/10., 1/10.],
                             [1/10., 2/10., 1/10.],
                             [1/10., 1/10., 1/10.]])
    mean_filter3 = np.array([[1/16., 2/16., 1/16.],
                             [2/16., 4/16., 2/16.],
                             [1/16., 2/16., 1/16.]])

    enhancement_filter1 = np.array([[-1., -1., -1.],
                                    [-1., 9., -1.],
                                    [-1., -1., -1.]])
    enhancement_filter2 = np.array([[1., -1., 1.],
                                    [-2., 5., -2.],
                                    [1., -2., 1.]])
    enhancement_filter3 = np.array([[0., -2., 0.],
                                    [-1., 5., -1.],
                                    [0., -1., 0.]])

    filter_dict = {}
    filter_dict['1'] = (mean_filter1, 'mean_filter_1')
    filter_dict['2'] = (mean_filter2, 'mean_filter_2')
    filter_dict['3'] = (mean_filter3, 'mean_filter_3')
    filter_dict['4'] = (enhancement_filter1, 'enhancement_filter_1')
    filter_dict['5'] = (enhancement_filter2, 'enhancement_filter_2')
    filter_dict['6'] = (enhancement_filter3, 'enhancement_filter_3')

    return filter_dict

# -----------------------------------------------------------------------------

def get_neighborhood(image, row, col, mode):
    """Return a 3x3 or 5x5 2D array of the local neighborhood pixels
    surrounding the given row and column.

    Parameters
    ----------
    image : 2D array
        The image from which to grab the neighborhood pixels.
    row : int
        The row number of the central pixel.
    col : int
        The column number of the central pixel.
    mode : string
        Can either be '3x3' to return a 3x3 neighborhood, or '5x5' to
        return a 5x5 neighborhood.

    Returns
    -------
    neighborhood : 2D array
        A 2D array of pixel values surrounding the central row/col
        pixel in the image.
    """

    if mode == '3x3':
        neighborhood = np.array([[image[row-1,col-1], image[row-1,col], image[row-1, col+1]],
                                 [image[row,col-1], image[row,col], image[row,col+1]],
                                 [image[row+1,col-1], image[row+1,col], image[row+1, col+1]]])

    elif mode == '5x5':
        neighborhood = np.array([[image[row-2,col-2], image[row-2,col-1], image[row-2,col], image[row-2,col+1], image[row-2,col+2]],
                                 [image[row-1,col-2], image[row-1,col-1], image[row-1,col], image[row-1,col+1], image[row-1,col+2]],
                                 [image[row,col-2], image[row,col-1], image[row,col], image[row,col+1], image[row,col+2]],
                                 [image[row+1,col-2], image[row+1,col-1], image[row+1,col], image[row+1,col+1], image[row+1,col+2]],
                                 [image[row+2,col-2], image[row+2,col-1], image[row+2,col], image[row+2,col+1], image[row+2,col+2]]])

    return neighborhood

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

if __name__ == '__main__':

    # Ask user what they want to do
    menu = '\nPlease select an option from the menu:\n'
    menu += '\t(1) Select pre-defined mask\n'
    menu += '\t(2) Provide a 3x3 mask\n'
    menu += '\t(3) Provide a 5x5 mask\n'
    response = raw_input(menu)
    assert response in ['1', '2', '3'], 'Not a valid response. Please try again.'

    # For pre-defined masks
    if response == '1':
        filter_dict = get_filter_dict()
        menu = '\nPlease select a mask:\n'
        menu += '\n(1) Mean filter 1: \n{}\n'.format(filter_dict['1'][0])
        menu += '\n(2) Mean filter 2: \n{}\n'.format(filter_dict['2'][0])
        menu += '\n(3) Mean filter 3: \n{}\n'.format(filter_dict['3'][0])
        menu += '\n(4) Enhancement filter 1: \n{}\n'.format(filter_dict['4'][0])
        menu += '\n(5) Enhancement filter 2: \n{}\n'.format(filter_dict['5'][0])
        menu += '\n(6) Enhancement filter 3: \n{}\n'.format(filter_dict['6'][0])
        choice = raw_input(menu)
        assert choice in ['1', '2', '3', '4', '5', '6'], 'Not a valid choice.  Please try again.'
        apply_mask(filter_dict[choice])

    # For user-suppled 3x3 mask
    if response == '2':
        prompt = '\nPlease enter 9 elements in order from left to right, top to bottom,\n'
        prompt += '\neach separated by a comma (e.g. "-1, -1, -1, -1, 9, -1, -1, -1, -1"):\n'
        mask_elements = raw_input(prompt)
        mask_elements = [float(item) for item in mask_elements.split(',')]
        mask = convert_user_mask_to_array(mask_elements)
        apply_mask((mask, 'user_3x3_mask'))

    # For user-suppled 5x5 mask
    if response == '3':
        prompt = '\nPlease enter 25 elements in order from left to right, top to bottom,\n'
        prompt += '\neach separated by a comma (e.g. "-1, -2, 3, -2, -1, ..." etc.:):\n'
        mask_elements = raw_input(prompt)
        mask_elements = [float(item) for item in mask_elements.split(',')]
        mask = convert_user_mask_to_array(mask_elements)
        apply_mask((mask, 'user_5x5_mask'))
