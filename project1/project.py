#! /usr/bin/env python

import os
import warnings

from astropy.table import Table
import matplotlib.pyplot as plt
import numpy as np
from skimage.io import imread
from skimage.io import imsave
from skimage import util

warnings.filterwarnings("ignore")

# -----------------------------------------------------------------------------

def classify_object(data_dict):
    """
    """

    # Circles have cirulatiry close to 1
    if 0.8 <= data_dict['Circularity'] <= 1.2:
        classification = 1

    # Lines have relatively large perimeter and are non-circular
    elif data_dict['Perimeter'] / float(data_dict['Area']) > 0.3 \
    and (data_dict['Circularity'] < 0.8 or data_dict['Circularity'] > 1.2):
        classification = 2

    # Misc objects is everything else
    else:
        classification = 3

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

    circularity = 4 * np.pi * (area / float(perimeter)**2)
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

    # Plot histogram to find threshold
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.hist(image)
    plt.savefig('test_hist.jpg')

    # Binary threshold the image
    threshold = 185
    print '\tApplying binary threshold with value {}'.format(threshold)
    image[np.where(image < threshold)] = 0
    image[np.where(image >= threshold)] = 1

    # Save the binary image
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(image, cmap='gray')
    plt.savefig('test_threshold.jpg')
    print '\tBinary threshold image saved to test_treshold.jpg'

    # Mark the objects
    mark_image(image)

    # Save the marked image
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(image, cmap='gray')
    plt.savefig('test_marked.jpg')
    print '\tMarked image saved to test_marked.jpg'

    # The number of marked objects is the maximum value - 2
    num_objects = np.max(image) - 2
    print '\n\tThe number of objects found is {}\n'.format(num_objects)

    # Build table of statistics
    data_table = Table(
        names=['Object', 'Area', 'Diameter', 'Perimeter', 'Circularity', 'Classification'],
        dtype=[int, int, int, int, float, int])
    for object_number in range(2, num_objects+3):
        data_dict = {}
        data_dict['Object'] = object_number
        data_dict['Area'] = find_area(image, object_number)
        data_dict['Diameter'] = find_diameter(data_dict['Area'])
        data_dict['Perimeter'] = find_perimeter(image, object_number)
        data_dict['Circularity'] = find_circularity(data_dict['Area'], data_dict['Perimeter'])
        data_dict['Classification'] = classify_object(data_dict)
        data_table.add_row(data_dict)
    print '\nResults:\n\n{}'.format(data_table)
    data_table.write('results.dat', format='ascii')
    print '\n\tResults file written to results.dat'

    # Make image of classified objects
    classified_image = np.copy(image)
    classified_image[classified_image == 1] = 0
    for entry in data_table:
        obj = np.where(classified_image == entry['Object'])
        classified_image[obj] = entry['Classification']

    # Make plot showing classified objects
    fig = plt.figure()
    ax = fig.add_subplot(111)
    im = ax.imshow(classified_image, interpolation='none')
    fig.colorbar(im, ax=ax)
    plt.savefig('test_classified.jpg')
    print '\tClassification image saved to test_classified.jpg'
