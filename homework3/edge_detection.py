#! /usr/bin/env python

"""Apply the Sobel edge detection algorithm on the given image.

This program will allow the user to supply a grayscale image to test
the Sobel edge detection algorithm on.

Author:
    Matthew Bourque, October, 2016

Use:
    The user can exectute this program from the command line as such:

        >>> python edge_detection.py

    Doing so will run the program with the pre-defined test image
    "cam.bmp".

    The user may also import the sobel() function from within the
    python, and pass it an image file, as such:

        from edge_detection import sobel()
        sobel('my_image.bmp')

Outputs:
    <image_file>_sobel.bmp - The image with the Sobel edge detection
    applied.

Dependencies:
    The user must have a Python 2.7 installion.  The numpy and skimage
    external libraries are also required.
"""

import numpy as np
from skimage.io import imread
from skimage.io import imsave

# -----------------------------------------------------------------------------

def get_neighborhood(image, row, col):
    """Return a 3x3 2D array of the local neighborhood pixels
    surrounding the given row and column.

    Parameters
    ----------
    image : 2D array
        The image from which to grab the neighborhood pixels.
    row : int
        The row number of the central pixel.
    col : int
        The column number of the central pixel.

    Returns
    -------
    neighborhood : 2D array
        A 2D array of pixel values surrounding the central row/col
        pixel in the image.
    """

    neighborhood = np.array([[image[row-1,col-1], image[row-1,col], image[row-1, col+1]],
                             [image[row,col-1], image[row,col], image[row,col+1]],
                             [image[row+1,col-1], image[row+1,col], image[row+1, col+1]]])

    return neighborhood

# -----------------------------------------------------------------------------

def get_sobel_masks():
    """Return the 3x3 mask arrays corresponding to the Sobel virtical
    and horizontal masks.

    Returns
    -------
    vertical_mask : 2D array
        The virtical Sobel mask.
    horizontal_mask : 2D array
        The horizontal Sobel mask.
    """

    vertical_mask = np.array([[-1., 2., -1.],
                              [0., 0., 0.],
                              [1., 2., 1.]])

    horizontal_mask = np.array([[-1., 0., 1.],
                                [-2., 0., 2.],
                                [-1., 0., 1.]])

    return vertical_mask, horizontal_mask

# -----------------------------------------------------------------------------

def sobel(image_file):
    """Apply the Sobel edge detection alrogithm on the given image.

    Parameters
    ----------
    image_file : string
        The path to the image file.
    """

    # Test image
    image = imread(image_file)

    # Initializations
    vertical_mask, horizontal_mask = get_sobel_masks()
    nrows = image.shape[0] - 1
    ncols = image.shape[1] - 1
    sobel_magnitude = np.copy(image.astype(float))
    sobel_magnitude_with_roberts = np.copy(image.astype(float))

    # Iterate over the image
    for row in range(1,nrows):
        for col in range(1,ncols):

            neighborhood = get_neighborhood(image, row, col)

            # Convolve the neighborhood with the Sobel mask
            vertical_solution = np.sum(neighborhood * vertical_mask)
            horizontal_solution = np.sum(neighborhood * vertical_mask)

            # Calculate the Sobel magnitude
            sobel_mag = np.sqrt(vertical_solution**2 + horizontal_solution**2)

            # Replace the pixel with the Sobel magnitude
            sobel_magnitude[row,col] = sobel_mag

    # Remap the Sobel magnitude with Roberts
    for row in range(1,nrows):
        for col in range(1,ncols):
            roberts = np.abs(sobel_magnitude[row,col] - sobel_magnitude[row-1,col-1]) + \
                      np.abs(sobel_magnitude[row,col-1] - sobel_magnitude[row-1,col])
            sobel_magnitude_with_roberts[row,col] = roberts

    # Normalize the images to -1, 1 (required by skimage)
    sobel_magnitude_with_roberts /= np.max(np.abs(sobel_magnitude_with_roberts))

    # Write out new image
    imsave(image_file.replace('.bmp', '_sobel.bmp'), sobel_magnitude_with_roberts)
    print '\nImage saved to {}'.format(image_file.replace('.bmp', '_sobel.bmp'))

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

if __name__ == '__main__':

    # Image to test with
    image_file = 'cam.bmp'

    sobel(image_file)
