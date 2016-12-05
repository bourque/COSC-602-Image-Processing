#! /usr/bin/env python

"""Identify, classify, and remove cosmic rays in Hubble Space Telescope
(HST) Wide Field Camera 3 (WFC3) 'dark' images.

Dark images are images in which the camera shutter is closed.  As such,
the image contains only inherent dark current and high-energy cosmic
rays.  Such images are used to remove the dark current from scientific
WFC3 data.

This program performs several tasks:

    (1) Uses a marking algorithm to identify indivdual cosmic rays
    (2) Calculates the area of each identified cosmic ray
    (3) Classifies cosmic rays as 'small', 'medium', or 'large'
        based on the area of the cosmic ray
    (4) Dilates the marked cosmic ray in order to conservatively
        mark the outer edges of the cosmic ray (i.e. in case the
        binary threshold is not ideal)
    (5) Remove the dilated cosmic rays from the original image
    (6) Average-combine several images to produce a cosmic ray
        cleaned image to be used for proper calibration of WFC3 data

Authors:
    Matthew Bourque, December 2016
    Arielle Leone, December 2016

Use:
    This program is intended to be executed via the command line as
    such:
        >>> python cr_reject.py

Outputs:
    Executing this program will result in several outputs:
    (1) test_*.png/.fits - A subset of the original image that is used
        in this program.
    (2) binary_*.png/.fits - The binary thresholded images
    (3) marked_*.png/.fits - Images with individual cosmic rays marked
        with unique values
    (4) *.dat - Data files with stats/classification of each cosmic
        ray for each image
    (5) area_*.png - Histograms of cosmic ray area distributions for
        each image
    (6) dilated_*.png/.fits - Images with cosmic ray mask dilated
    (7) cleaned_*.png/.fits - Original images with cosmic rays removed
    (8) combined.png/.fits - The final combined, cr-cleaned image

Dependencies:

    The user must have a Python 3.5 installation.  The astropy,
    matplotlib and numpy external libraries are also required.
"""

import glob
import os

from astropy.io import ascii
from astropy.io import fits
from astropy.table import Table
import matplotlib.pyplot as plt
import numpy as np

# -----------------------------------------------------------------------------

def dilate_image(image):
    """Perform dilation on the given cosmic ray mask.

    Parameters
    ----------
    image : 2D array
        The image to dilate.

    Returns
    -------
    dilated_image : 2D array
        The dilated image.
    """

    print('Performing Dilation')

    # Initializations
    nrows = image.shape[0] - 1
    ncols = image.shape[1] - 1
    dilated_image = np.copy(image)

    mask = np.array([[1, 1, 1],
                     [1, 1, 1],
                     [1, 1, 1]])

    for row in range(2,nrows):
        print('\t{}% complete'.format(int((row/nrows) * 100)), end='\r')
        for col in range(2,ncols):

            if image[row, col] == 1:
                neighborhood = get_neighborhood(image, row, col)
                dilation_results = np.logical_or(mask,neighborhood).astype(np.int)
                dilated_image[row-1:row+2,col-1:col+2] = dilation_results

    return dilated_image

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

def get_range(index, max_index):
    """Return a range of values that is either index:index+50 or
    index:(edge of image).

    This function is used to determine the range of values in which
    to iterate over, as to not exceed the edge of the image.  A range
    of index:index+50 is attempted, but if the edge of the image is
    reached, then the range index:(edge of image) is returned.

    Parameters
    ----------
    index : int
        The index which starts the range.
    max_index : int
        The maximum allowable index (i.e. the edge of the iamge).

    Returns
    -------
    range() : iterable
        The range of indices to iterate over.
    """

    if index + 50 >= max_index:
        return range(index, max_index)
    else:
        return range(index, index+50)

# -----------------------------------------------------------------------------

def make_histograms(image):
    """Create plot of histogram and corresponding binary thresholded
    image for a range of thresholds.

    This funciton will create a plot showing the original image, a
    histogram showing the distribution of pixel values for the original
    image, and a binary thresholded image for a given threshold.  Plots
    are produced for thresholds between -10 and 30 electrons, with step
    size of 0.5 electrons.  These plots can be used to create an
    animated gif showing how the threshold applied affects the binary
    thresholded image.

    Parameters
    ----------
    image : 2D array
        The image to process.
    """

    print('Creating histograms')

    plt.style.use('bmh')
    plt.axis('off')
    threshold_image = np.copy(image)
    range_list = [i*0.5 for i in range(-20, 61)]
    count = 0

    for threshold in range_list:

        # Binary threshold the image
        threshold_image[np.where(image < threshold)] = 0
        threshold_image[np.where(image >= threshold)] = 1

        # Make histogram
        hist = np.histogram(image, bins=500, range=(-10,30), density=False)
        x = hist[1][:-1]
        y = hist[0]

        # Initialize the plot
        fig = plt.figure(figsize=(10,8))
        ax1 = plt.subplot2grid((2,2),(0,0), colspan=2)
        ax2 = plt.subplot2grid((2,2),(1,0))
        ax3 = plt.subplot2grid((2,2),(1,1))

        # Plot the histogram
        ax1.plot(x, y, color='k', linewidth=1.5)
        ax1.set_xlabel('Dark Current (e-)')
        ax1.set_ylabel('Number of Pixels (% of detector)')
        yticks=ax1.get_yticks().tolist()
        for i, ytick in enumerate(yticks):
            yticks[i] = round((ytick / (image.shape[0] * image.shape[1])) * 100, 2)
        ax1.set_yticklabels(yticks)
        ax1.axvline(threshold, color='r')

        # Plot the original image
        ax2.imshow(image, cmap='gray', vmin=-20, vmax=20)
        ax2.set_xticklabels([])
        ax2.set_yticklabels([])
        ax2.grid(b=False)

        # Plot the thresholded image
        ax3.imshow(threshold_image, cmap='gray')
        ax3.set_xticklabels([])
        ax3.set_yticklabels([])
        ax3.grid(b=False)

        # Save the image
        plt.savefig('histogram_plots/hist_thresh{}.png'.format(count))

        count += 1

# -----------------------------------------------------------------------------

def mark_image(image, data_table):
    """Mark each cosmic ray in the image with a unique value.

    The unique value to mark cosmic rays with starts with 2 (as to
    distinguish it from the background and other cosmic rays) and
    increases by 1 for each unique cosmic ray. During this function,
    the data_table is also updated with the cosmic ray number and its
    (first found) position.

    Parameters
    ----------
    image : 2D array
        The image in which the cosmic rays reside.
    data_table : table
        A table that is to hold results.

    """

    print('Marking objects')

    # Initializations
    nrows = image.shape[0] - 1
    ncols = image.shape[1] - 1
    value = 2

    # Mark the objects
    for row in range(0,nrows):
        print('\t{}% complete'.format(int((row/nrows) * 100)), end='\r')
        for col in range(0,ncols):

            if image[row, col] == 1:

                # Do not mark if it is just one pixel
                neighborhood = get_neighborhood(image, row, col)
                if np.count_nonzero(neighborhood) > 1:

                    # Mark the pixel
                    image[row,col] = value

                    # Record it for the data_table
                    data_dict = {'Object' : value - 1, 'Position' : (row,col)}
                    data_table.add_row(data_dict)

                    # Sweep right and down
                    row_range = get_range(row, nrows)
                    col_range = get_range(col, ncols)
                    for row_r in row_range:
                        for col_r in col_range:

                            if image[row_r,col_r] == value:

                                # Mark any connected pixels in the neighborhood
                                for row_index in [-1, 0, 1]:
                                    for col_index in [-1, 0, 1]:

                                        # There are neighboring pixels in object, mark with value
                                        if image[row_r + row_index, col_r + col_index] == 1:
                                            image[row_r + row_index, col_r + col_index] = value

                    # Sweep left and up
                    for row_l in reversed(row_range):
                        for col_l in reversed(col_range):

                            if image[row_l,col_l] == value:

                                # Mark any connected pixels in the neighborhood
                                for row_index in [-1, 0, 1]:
                                    for col_index in [-1, 0, 1]:

                                        # There are neighboring pixels in object, mark with value
                                        if image[row_l + row_index, col_l + col_index] == 1:
                                            image[row_l + row_index, col_l + col_index] = value

                    value += 1

    # Set the individual 1 pixels to 0
    image[np.where(image == 1)] = 0

# -----------------------------------------------------------------------------

def perform_statistics(image, data_table):
    """Perform statistics on each cosmic ray and update the data_table.

    This function will calculate the area for each cosmic ray and
    classify it as 'small', 'medium', or 'large' based on its area.

    Parameters
    ----------
    image : 2D array
        The image that contains the cosmic rays.
    data_table : astropy Table object
        The table that holds the statistics and classifications.
    """

    print('Calculating statistics')

    num_objects = int(np.max(data) - 1)

    for object_number in range(2, num_objects+2):

        # Determine the area
        area = image[np.where(image == object_number)].size
        data_table[object_number-2]['Area'] = area

        # Classify the cosmic ray
        if area < 5:
        	data_table[object_number-2]['Classification'] = 'S'
        elif 5 <= area <= 12:
        	data_table[object_number-2]['Classification'] = 'M'
        else:
        	data_table[object_number-2]['Classification'] = 'L'

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

if __name__ == '__main__':

    # Get list of files
    filenames = glob.glob('data/nn*_blv_tmp.fits')

    # Initialize master data table dict
    data_table_dict = {}

    for i, filename in enumerate(filenames):

        print('\n\nProcessing image {} of {}'.format(i+1, len(filenames)))
        rootname = os.path.basename(filename).split('_')[0]

        # Read in the image
        orig_data = fits.getdata(filename, 1)

        # Get subset of image to test with
        orig_data = orig_data[0:1000, 0:1000]
        data = np.copy(orig_data)

        # Save the test image
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.imshow(data, cmap='gray', vmin=-20, vmax=20)
        plt.savefig('data/test_{}.png'.format(rootname))
        if os.path.exists('data/test_{}.fits'.format(rootname)):
            os.remove('data/test_{}.fits'.format(rootname))
        hdu = fits.PrimaryHDU()
        hdu1 = fits.ImageHDU(data)
        hdulist = fits.HDUList([hdu,hdu1])
        hdulist.writeto('data/test_{}.fits'.format(rootname))

        # Make histogram movie to find best threshold
        #make_histograms(data[0:500,0:500])

        # Binary threshold the image
        threshold = 25
        data[np.where(data < threshold)] = 0
        data[np.where(data >= threshold)] = 1

        # Save the binary thresholded image
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.imshow(data, cmap='gray')
        plt.savefig('binary/binary_{}.png'.format(rootname))
        if os.path.exists('binary/binary_{}.fits'.format(rootname)):
            os.remove('binary/binary_{}.fits'.format(rootname))
        hdu = fits.PrimaryHDU()
        hdu1 = fits.ImageHDU(data)
        hdulist = fits.HDUList([hdu,hdu1])
        hdulist.writeto('binary/binary_{}.fits'.format(rootname))

        # Initialize a data table that will hold results
        data_table = Table(
        names=['Object', 'Position', 'Area', 'Classification'],
        dtype=[int, tuple, int, str])

        # Mark the cosmic rays
        mark_image(data, data_table)
        marked_image = np.copy(data)

        # Save the marked image
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.imshow(data, cmap='gray')
        plt.savefig('marked/marked_{}.png'.format(rootname))
        if os.path.exists('marked/marked_{}.fits'.format(rootname)):
            os.remove('marked/marked_{}.fits'.format(rootname))
        hdu = fits.PrimaryHDU()
        hdu1 = fits.ImageHDU(data)
        hdulist = fits.HDUList([hdu,hdu1])
        hdulist.writeto('marked/marked_{}.fits'.format(rootname))

        # Perform statistics/classification
        perform_statistics(data, data_table)
        data_table_dict[rootname] = data_table

        # Write out data table
        ascii.write(data_table, '{}.dat'.format(rootname))

	    # Plot the distribution of area
        plt.style.use('bmh')
        plt.axis('off')
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.hist(data_table['Area'], bins=40, range=(0,40), color='g')
        ax.text(27, 450, '{} total'.format(len(data_table['Area'])), color='g')
        ax.set_xlabel('Area (pixels)')
        plt.savefig('area/area_{}.png'.format(rootname))

    	# Plot the classified cosmic rays
        classified_image = np.copy(data)
        for entry in data_table:
            cr = np.where(classified_image == entry['Object'] + 1)
            if entry['Classification'] == 'S':
                classification_value = 1
            elif entry['Classification'] == 'M':
                classification_value = 2
            else:
                classification_value = 3
            classified_image[cr] = classification_value
        fig = plt.figure()
        ax = fig.add_subplot(111)
        im = ax.imshow(classified_image, interpolation='none')
        cbar = fig.colorbar(im, ax=ax, ticks=[1, 2, 3])
        cbar.ax.set_yticklabels(['Small', 'Medium', 'Large'])
        plt.savefig('classified/classified_{}.png'.format(rootname))

        # Set marked pixels back to 1
        data[np.where(data > 0)] = 1

        # Perform dilation
        dilated_image = dilate_image(data)

        # Save the dilated image
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.imshow(dilated_image, cmap='gray')
        plt.savefig('dilated/dilated_{}.png'.format(rootname))
        if os.path.exists('dilated/dilated_{}.fits'.format(rootname)):
            os.remove('dilated/dilated_{}.fits'.format(rootname))
        hdu = fits.PrimaryHDU()
        hdu1 = fits.ImageHDU(dilated_image)
        hdulist = fits.HDUList([hdu,hdu1])
        hdulist.writeto('dilated/dilated_{}.fits'.format(rootname))

        # Remove the CRs from the original image
        orig_data[np.where(dilated_image == 1)] = 0

        # Save the CR-cleaned image
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.imshow(orig_data, cmap='gray')
        plt.savefig('cleaned/cleaned_{}.png'.format(rootname))
        if os.path.exists('cleaned/cleaned_{}.fits'.format(rootname)):
            os.remove('cleaned/cleaned_{}.fits'.format(rootname))
        hdu = fits.PrimaryHDU()
        hdu1 = fits.ImageHDU(orig_data)
        hdulist = fits.HDUList([hdu,hdu1])
        hdulist.writeto('cleaned/cleaned_{}.fits'.format(rootname))

    # Stack the cleaned images
    print('Combining images')
    cleaned_images = glob.glob('cleaned/cleaned_*.fits')
    image_stack = []
    for image in cleaned_images:
        with fits.open(image) as hdulist:
            data = hdulist[1].data
            image_stack.append(data)
    combined_image = np.mean(image_stack, axis=0)

    # Save the combined image
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(combined_image, cmap='gray', vmin=-20, vmax=20)
    plt.savefig('combined.png')
    if os.path.exists('combined.fits'):
        os.remove('combined.fits')
    hdu = fits.PrimaryHDU()
    hdu1 = fits.ImageHDU(combined_image)
    hdulist = fits.HDUList([hdu,hdu1])
    hdulist.writeto('combined.fits')