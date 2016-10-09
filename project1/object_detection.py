#! /usr/bin/env python

"""Mark and classify objects in the test image for COSC 602 project 1.

This program will mark and classify unique objects in the given image
for COSC 602 Project 1 (renamed to "test.bmp").  To mark objects, the
image is iterated over using 8-connectivity to identify unique objects
and each object is marked with a unique value.  Once marked, various
statistics such as area and perimeter are calculated for each object.
Each object is then classified into one of three categories based on
these statistics: (1) circular object (value=1), (2) line-like object
(value=2), and miscellanous object (value=3).  Intermediate and final
images and results are saved to the working directory.

Authors:
    Matthew Bourque, October 2016
    Arielle Leone, October 2016

Use:
    This program is intended to be executed via the command line as
    such:

        >>> python object_detection.py

Outputs:
    Executing this program will result in several outputs:

    (1) test_hist.jpg - A histogram of the pixel values in test.bmp,
            used for determing a good binary threshold.
    (2) test_threshold.jpg - The binary thresholded image.
    (3) test_marked.jpg - The image with each object given a unique
            value.
    (4) results.dat - A file contaning statistics for each object (i.e.
            object value, area, diameter, perimeter, circularity, and
            classification number).
    (5) test_classified.jpg - An image with each object having a value
            of its corresponding classifcation number.

Dependencies:
    The user must have a Python 2.7 installation.  The astropy,
    matplotlib, numpy, and skimage external libraries are also
    required.
"""

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
    """Classify each object into one of three categories: (1) circular
    objects (value=1), line-like objects (value=2), and miscellanous
    objects (value=3).

    Parameters
    ----------
    data_dict : dictionary
        A dictionary whose keys are the object statistics (e.g. Area,
        Perimeter, etc.) and whose values are the values for those
        statistics.

    Returns
    -------
    classification : int
        The classification for the object.  1 is for circular objects,
        2 is for line-like objects, 3 is for oddly shaped,
        miscellanous objects.
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
    """Return the area of the object.

    Parameters
    ----------
    image : 2D array
        The image in which the object is located.
    object_number : int
        The pixel value of the object.

    Returns
    -------
    area : int
        The area of the object.
    """

    area = image[np.where(image == object_number)].size

    return area

# -----------------------------------------------------------------------------

def find_circularity(area, perimeter):
    """Return the circularity of the object.  The circularity is
    defined as 4 * pi * area divided py the perimeter squared.

    Parameters
    ----------
    area : int
        The area of the object.
    perimeter : int
        The perimeter of the object.

    Returns
    -------
    circularity : float
        The circularity of the object.
    """

    circularity = 4 * np.pi * (area / float(perimeter)**2)
    circularity = round(circularity, 3)

    return circularity

# -----------------------------------------------------------------------------

def find_diameter(area):
    """Return the diameter of the object.

    Parameters
    ----------
    area : int
        The area of the object.

    Returns
    -------
    diameter : int
        The diameter of the object.
    """

    diameter = np.sqrt(area / np.pi) * 2
    diameter = int(round(diameter, 0))

    return diameter

# -----------------------------------------------------------------------------

def find_perimeter(image, object_number):
    """Return the perimeter of the object.  The perimeter in this case
    is defined as the number of pixels that have a '0' as a 4-connected
    neighbor (i.e. doesn't have a neighboring pixel that is part of the
    object itself).

    Parameters
    ----------
    image : 2D array
        The image in which the object is located.
    object_number : int
        The pixel value of the object.

    Returns
    -------
    perimeter : int
        The perimeter of the object.
    """

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
    """Mark each object in the image with a unique value.

    Since the object pixels have a value of 0 and the 'background'
    pixels have a value of 1, the unique value to mark objects with
    starts with 2 (as to distinguish it from the background and other
    objects) and increases by 1 for each unique object.

    Parameters
    ----------
    image : 2D array
        The image in which the objects reside.
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
