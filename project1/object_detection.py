#! /usr/bin/env python

"""Mark and classify objects in the test image for COSC 602 project 1.

This program will mark and classify unique objects in the given image
for COSC 602 Project 1 (renamed to "test.bmp").  To mark objects, the
image is iterated over using 8-connectivity to identify unique objects
and each object is marked with a unique value.  Once marked, various
statistics such as area and perimeter are calculated for each object.
Each object is then classified into one of three categories based on
these statistics: (1) small, (2) medium, and (3) large.  Intermediate
and final images and results are saved to the working directory.

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

warnings.filterwarnings("ignore")

# -----------------------------------------------------------------------------

def classify_objects(data_table):
    """Classify each object into one of three categories (small,
    medium, or large) based on the object's area, and update the data
    table with the result.

    Parameters
    ----------
    data_dict : dictionary
        A dictionary whose keys are the object statistics (e.g. Area,
        Perimeter, etc.) and whose values are the values for those
        statistics.
    """

    max_area = max(data_table['Area'])

    for obj in data_table:
        if obj['Area'] <= (max_area / 3.):
            obj['Classification'] = 'Small'
        elif (max_area / 3.) < obj['Area'] < (2 * max_area / 3.):
            obj['Classification'] = 'Medium'
        else:
            obj['Classification'] = 'Large'

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
    circularity = int(round(circularity*100, 0))

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
    is defined as the number of pixels that have a '0' as a 8-connected
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

                # Check to see if an 8-connected nieghbor contains a 1
                neighbors = [
                    image[row-1,col-1], image[row-1,col], image[row-1,col+1],
                    image[row,col-1], image[row,col], image[row,col+1],
                    image[row+1,col-1], image[row+1,col], image[row+1,col+1]]
                if 1 in neighbors:
                    perimeter += 1

    return perimeter

# -----------------------------------------------------------------------------

def mark_image(image, data_table):
    """Mark each object in the image with a unique value.

    Since the object pixels have a value of 0 and the 'background'
    pixels have a value of 1, the unique value to mark objects with
    starts with 2 (as to distinguish it from the background and other
    objects) and increases by 1 for each unique object.

    During this function, the data_table is also updated with the
    object number and its (first found) position.

    Parameters
    ----------
    image : 2D array
        The image in which the objects reside.
    data_table : table
        A table that is to hold results.
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

                # Update the data table with object number and position
                data_dict = {'Object' : value - 1, 'Position' : (row,col)}
                data_table.add_row(data_dict)

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
    threshold = 175
    print '\tApplying binary threshold with value {}'.format(threshold)
    image[np.where(image < threshold)] = 0
    image[np.where(image >= threshold)] = 1

    # Save the binary image
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(image, cmap='gray')
    plt.savefig('test_threshold.jpg')
    print '\tBinary threshold image saved to test_treshold.jpg'

    # Initialize a data table that will hold results
    data_table = Table(
        names=['Object', 'Position', 'Area', 'Diameter', 'Perimeter', 'Circularity', 'Classification'],
        dtype=[int, tuple, int, int, int, int, str])

    # Mark the objects
    mark_image(image, data_table)

    # Save the marked image
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(image, cmap='gray')
    plt.savefig('test_marked.jpg')
    print '\tMarked image saved to test_marked.jpg'

    # The number of marked objects is the maximum value - 2
    num_objects = np.max(image) - 1
    print '\n\tThe number of objects found is {}\n'.format(num_objects)

    # Update the data table with statistics
    print '\tCalculating statistics'
    for object_number in range(2, num_objects+2):
        data_table[object_number-2]['Area'] = find_area(image, object_number)
        data_table[object_number-2]['Diameter'] = find_diameter(data_table[object_number-2]['Area'])
        data_table[object_number-2]['Perimeter'] = find_perimeter(image, object_number)
        data_table[object_number-2]['Circularity'] = find_circularity(data_table[object_number-2]['Area'], data_table[object_number-2]['Perimeter'])

    # Classify each object
    print '\tClassifying objects'
    classify_objects(data_table)

    # Make image of classified objects
    classified_image = np.copy(image)
    classified_image[classified_image == 1] = 0
    for entry in data_table:
        obj = np.where(classified_image == entry['Object'] + 1)
        if entry['Classification'] == 'S':
            classification_value = 1
        elif entry['Classification'] == 'M':
            classification_value = 2
        else:
            classification_value = 3
        classified_image[obj] = classification_value

    # Print out report of results
    print '\nResults:\n\n{}'.format(data_table)
    data_table.write('results.dat', format='ascii.fixed_width')
    print '\n\tResults file written to results.dat'

    # Make plot showing classified objects
    fig = plt.figure()
    ax = fig.add_subplot(111)
    im = ax.imshow(classified_image, interpolation='none')
    cbar = fig.colorbar(im, ax=ax, ticks=[1, 2, 3])
    cbar.ax.set_yticklabels(['Small', 'Medium', 'Large'])
    plt.savefig('test_classified.jpg')
    print '\tClassification image saved to test_classified.jpg'
