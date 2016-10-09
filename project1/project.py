#! /usr/bin/env python

import os

from astropy.io import fits
from astropy.io import ascii
from astropy.table import Table
import matplotlib.pyplot as plt
import numpy as np
from skimage.io import imread
from skimage.io import imsave

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

    return circularity

# -----------------------------------------------------------------------------

def find_diameter():
    """

    """

    diameter = 1

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

    # Initializations
    nrows = image.shape[0]
    ncols = image.shape[1]
    value = 2

    # Mark the objects
    for row in range(0,nrows):
        for col in range(0,ncols):

            print 'Checking ({},{})'.format(row,col)

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

    image_file = 'test.bmp'
    image = imread(image_file)

    # Plot histogram to find threshold
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.hist(image)
    plt.savefig('test_hist.png')

    # Binary threshold the image
    threshold = 185
    image[np.where(image < threshold)] = 0
    image[np.where(image >= threshold)] = 1

    # Save the binary image
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(image, cmap='gray')
    plt.savefig('test_threshold.png')

    # Mark the objects
    mark_image(image)

    # Save the marked image
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(image, cmap='gray')
    plt.savefig('test_marked.png')

    if os.path.exists('image.fits'):
        os.remove('image.fits')
    hdu = fits.PrimaryHDU()
    hdu1 = fits.ImageHDU(image)
    hdulist = fits.HDUList([hdu, hdu1])
    hdulist.writeto('image.fits')

    # The number of marked objects is the maximum value - 2
    num_objects = np.max(image) - 2
    print 'The number of objects found is {}'.format(num_objects)

    # Build table of statistics
    data_table = Table(names=['Object #', 'Area', 'Diameter', 'Perimeter', 'Circularity', 'Classification #'])
    for object_number in range(2, num_objects+2):
        data_dict = {}
        data_dict['Object #'] = object_number
        data_dict['Area'] = find_area(image, object_number)
        data_dict['Diameter'] = find_diameter()
        data_dict['Perimeter'] = find_perimeter(image, object_number)
        data_dict['Circularity'] = find_circularity(data_dict['Area'], data_dict['Perimeter'])
        data_table.add_row(data_dict)
    print data_table

    # Classify the objects
