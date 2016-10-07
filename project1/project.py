#! /usr/bin/env python

from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
from skimage.io import imread
from skimage.io import imsave

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

if __name__ == '__main__':

    image_file = 'test.bmp'
    image = imread(image_file)

    # # Plot histogram to find threshold
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.hist(image)
    # plt.savefig('test_hist.png')

    # Binary threshold the image
    threshold = 185
    image[np.where(image < threshold)] = 0
    image[np.where(image >= threshold)] = 1

    # # Save the binary image
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.imshow(image, cmap='gray')
    # plt.savefig('test_threshold.png')

    # Initializations
    nrows = image.shape[0]
    ncols = image.shape[1]
    marked_image = np.zeros(image.shape)
    value = 2

    # Mark the objects
    for row in range(0,nrows):
        for col in range(0,ncols):

            print 'Checking ({},{})'.format(row,col)

            if image[row, col] == 0:

                # Mark the pixel
                image[row,col] = value

                for row in range(0,nrows):
                    for col in range(0,ncols):

                        if image[row,col] == value:

                            # Mark any connected pixels in the neighborhood
                            for row_index in [-1, 0, 1]:
                                for col_index in [-1, 0, 1]:

                                    # There are neighboring pixels in object, mark with value
                                    if image[row + row_index, col + col_index] == 0:
                                        image[row + row_index, col + col_index] = value

                value += 1

    # Save the marked image
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(image, cmap='gray')
    plt.savefig('test_marked.png')

    # The number of marked objects is the maximum value - 2
    num_objects = np.max(image) - 2
    print 'The number of objects found is {}'.format(num_objects)

    # hdu = fits.PrimaryHDU()
    # hdu1 = fits.ImageHDU(image)
    # hdulist = fits.HDUList([hdu, hdu1])
    # hdulist.writeto('image.fits')
