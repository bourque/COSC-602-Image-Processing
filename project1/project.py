#! /usr/bin/env python

import os

from astropy.io import fits
from astropy.io import ascii
from astropy.table import Table
import matplotlib.pyplot as plt
import numpy as np
from skimage.io import imread
from skimage.io import imsave
from skimage import util

import warnings
warnings.filterwarnings("ignore")

# -----------------------------------------------------------------------------

def classify_object(data_dict):
    """
    """

    print data_dict['Object #'], data_dict['Diameter'] / float(data_dict['Area'])

    # Lines have relatively large diameter and are non-circular
    if data_dict['Perimeter'] / float(data_dict['Area']) > 0.3 \
    and (data_dict['Circularity'] < 0.5 or data_dict['Circularity'] > 1.5):
        classification = 'Line'

    # Circles have cirulatiry close to 1
    elif 0.5 <= data_dict['Circularity'] <= 1.5:
        classification = 'Circle'

    # Misc objects is everything else
    else:
        classification = 'Misc'

    return classification

# -----------------------------------------------------------------------------

def find_area(image, object_number):
    """

    """

    area = image[np.where(image == object_number)].size

    return area

# -----------------------------------------------------------------------------

def find_circularity(area, perimeter):
    """

    """

    circularity = (4 * np.pi * area) / float(perimeter)**2
    circularity = round(circularity, 3)

    return circularity

# -----------------------------------------------------------------------------

def find_diameter(area):
    """

    """

    diameter = np.sqrt(area / np.pi) * 2
    diameter = int(round(diameter, 0))

    return diameter

# -----------------------------------------------------------------------------

def find_perimeter(image, object_number):
    """

    """

    # The permieter is the number of '1' pixels that have '0' as neighbors
    nrows = image.shape[0]
    ncols = image.shape[1]
    perimeter = 0

    for row in range(0,nrows):
        for col in range(0,ncols):

            if image[row,col] == object_number:

                # Check to see if nieghbors contain 0
                neighbors = [image[row-1,col], image[row,col+1], image[row+1,col], image[row,col-1]]
                if 1 in neighbors:
                    perimeter += 1

    return perimeter

# -----------------------------------------------------------------------------

def mark_image(image):
    """
    """

    print '\tMarking objects'

    # Initializations
    nrows = image.shape[0]
    ncols = image.shape[1]
    value = 2

    # Mark the objects
    for row in range(0,nrows):
        for col in range(0,ncols):

            if image[row, col] == 0:

                # Mark the pixel
                image[row,col] = value

                # Sweep right and down
                for row in range(0,nrows):
                    for col in range(0,ncols):

                        if image[row,col] == value:

                            # Mark any connected pixels in the neighborhood
                            for row_index in [-1, 0, 1]:
                                for col_index in [-1, 0, 1]:

                                    # There are neighboring pixels in object, mark with value
                                    if image[row + row_index, col + col_index] == 0:
                                        image[row + row_index, col + col_index] = value


                # Sweep left and up
                for row in reversed(range(0,nrows)):
                    for col in reversed(range(0,ncols)):

                        if image[row,col] == value:

                            # Mark any connected pixels in the neighborhood
                            for row_index in [-1, 0, 1]:
                                for col_index in [-1, 0, 1]:

                                    # There are neighboring pixels in object, mark with value
                                    if image[row + row_index, col + col_index] == 0:
                                        image[row + row_index, col + col_index] = value

                value += 1


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

if __name__ == '__main__':

    # Read in image file
    image_file = 'test.bmp'
    print '\tReading in image {}'.format(image_file)
    image = imread(image_file)

    # inverted = util.invert(image)
    # imsave('test_invert.bmp', inverted)

    # Plot histogram to find threshold
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.hist(image)
    plt.savefig('test_hist.png')

    # Binary threshold the image
    threshold = 185
    print '\tApplying binary threshold with value {}'.format(threshold)
    image[np.where(image < threshold)] = 0
    image[np.where(image >= threshold)] = 1

    # Save the binary image
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(image, cmap='gray')
    plt.savefig('test_threshold.png')
    print '\tBinary threshold image saved to test_treshold.png'

    # Mark the objects
    mark_image(image)

    # Save the marked image
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(image, cmap='gray')
    plt.savefig('test_marked.png')
    print '\tMarked image saved to test_marked.png'

    if os.path.exists('image.fits'):
        os.remove('image.fits')
    hdu = fits.PrimaryHDU()
    hdu1 = fits.ImageHDU(image)
    hdulist = fits.HDUList([hdu, hdu1])
    hdulist.writeto('image.fits')

    # The number of marked objects is the maximum value - 2
    num_objects = np.max(image) - 2
    print '\n\tThe number of objects found is {}\n'.format(num_objects)

    # Build table of statistics
    data_table = Table(
        names=['Object #', 'Area', 'Diameter', 'Perimeter', 'Circularity', 'Classification'],
        dtype=[int, int, int, int, float, str])
    for object_number in range(2, num_objects+3):
        data_dict = {}
        data_dict['Object #'] = object_number
        data_dict['Area'] = find_area(image, object_number)
        data_dict['Diameter'] = find_diameter(data_dict['Area'])
        data_dict['Perimeter'] = find_perimeter(image, object_number)
        data_dict['Circularity'] = find_circularity(data_dict['Area'], data_dict['Perimeter'])

        # Classify the object
        data_dict['Classification'] = classify_object(data_dict)

        data_table.add_row(data_dict)
    print '\nResults:\n\n{}'.format(data_table[:])
