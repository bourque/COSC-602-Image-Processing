#! /usr/bin/env python

"""Perform the Hough transform on a given image.

Authors:
    Matthew Bourque, October 2016

Use:
    This program can be executed via the command line as such:

        >>> python hough_transform.py

    In this case, the user will be prompted to supply the angles of
    interest and the minimum number of pixel per line parameters.  The
    user can also execute the program from within the python
    environment as such:

        from hough_transform import hough
        hough(image_file, angles, pix_per_line)

    where <image_file> is the file to process, <angles> is the angles
    interest (e.g. (0,180)), and <pix_per_line> is the minimum number
    of pixels per line (e.g. 10).  The user may also supply addition
    arguments:

        from hough_transform import hough
        hough(image_file, angles, pix_per_line, delta_theta, delta_p, threshold)

    where <delta_theta> is the change in theta to increment by (i.e.
    the size of the theta block in the rho/theta space) (e.g. 1.0),
    <delta_p> is the block size of rho in the rho/theta space, (e.g.
    1.0) and <threshold> is the threshold to apply to the image file
    (e.g. 100), above which will be considered a part of a line in the
    image.

    An example call would be the following:

        hough('cam_sobel.bmp', (0,180), 10, 1.0, 1.0, 100)

Output:
    <image_file>_hough.bmp - The image with the Hough transform
    applied.

Dependencies:
    The user must have a Python 2.7 installion.  The numpy and skimage
    external libraries are also required.
"""

import numpy as np
from skimage.io import imread
from skimage.io import imsave

# -----------------------------------------------------------------------------

def hough(image_file, angles, pix_per_line, delta_theta=1.0, delta_p = 1.0, threshold=100):
    """Perform the Hough transform on the given image_file with the
    given parameters.

    Parameters
    ----------
    image_file : string
        The path to the image to process.
    angles : tuple
        A tuple containing the angles of interst (e.g. (0,180)).
    pix_per_line : int
        The minimum number of pixels per line threshold that is applied
        to the transformed image.
    delta_theta (optional) : float
        The change in theta to increment by while performing the Hough
        transform (i.e. the theta block size in the rho/theta space).
    delta_p (optional) : float
        The block size of rho in the rho/theta space.
    threshold (optional) : int
        The threshold to apply to the image file, above which will be
        considered a part of a line in the image.
    """

    # Open the image file
    image = imread(image_file)

    # Initializations
    nrows = image.shape[0]
    ncols = image.shape[1]

    # Initialize transformed image
    rho = np.arange(0.0, (np.sqrt(2)*nrows)+1, delta_p)
    theta = np.arange(float(angles[0]), float(angles[1])+1, delta_theta)
    transformed_image = np.zeros((len(rho), len(theta))).astype(np.uint)

    # Perform the Hough transform
    for row in range(0,nrows):
        for col in range(0,ncols):
            if image[row,col] >= threshold:
                theta = angles[0]
                while(theta<=angles[1]):
                    p = np.abs(row*np.cos(theta*(np.pi/180.)) + col*np.sin(theta*(np.pi/180.)))
                    p = round(p, 1)
                    transformed_image[p, theta] += 1
                    theta += delta_theta

    # Threshold the resulting image for number of pixels per line:
    transformed_image[np.where(transformed_image >= pix_per_line)] = 255
    transformed_image[np.where(transformed_image < pix_per_line)] = 0

    # Save the image
    imsave(image_file.replace('.bmp', '_hough.bmp'), transformed_image)
    print 'Image saved to {}'.format(image_file.replace('.bmp', '_hough.bmp'))

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

if __name__ == '__main__':

    # Get the angles of interest from user
    prompt='\nPlease enter the line angle(s) of interest (e.g. "0-180"):\n'
    angle_of_interest = raw_input(prompt)
    assert '-' in angle_of_interest, 'Not a valid range. Please try again.'
    angle_of_interest = (float(angle_of_interest.split('-')[0]), float(angle_of_interest.split('-')[1]))
    assert 0 <= angle_of_interest[0] < 180, 'Not a valid range. Please try again.'
    assert 0 < angle_of_interest[1] <= 180, 'Not a valid range. Please try again.'

    # Get the minimum number of pixels per line from user
    prompt='\nPlease enter the minimum number of pixels per line (e.g. "10"):\n'
    pix_per_line = int(raw_input(prompt))

    # Perform the transform
    hough('cam_sobel.bmp', angle_of_interest, pix_per_line)