import os

from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np

# -----------------------------------------------------------------------------

def dilate_image(image):
    """
    """

    print('\nPerforming Dilation')

    # Initializations
    nrows = image.shape[0] - 1
    ncols = image.shape[1] - 1
    dilated_image = np.copy(image)

    mask = np.array([[0, 1, 0],
                     [1, 1, 1],
                     [0, 1, 0]])

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

    if index + 50 >= max_index:
        return range(index, max_index)
    else:
        return range(index, index+50)

# -----------------------------------------------------------------------------

def mark_image(image):
    """
    """

    print('\nMarking objects')

    # Initializations
    nrows = image.shape[0] - 1
    ncols = image.shape[1] - 1
    value = 2

    # Mark the objects
    for row in range(0,nrows):
        print('\t{}% complete'.format(int((row/nrows) * 100)), end='\r')
        for col in range(0,ncols):

            if image[row, col] == 1:

                # Mark the pixel
                image[row,col] = value

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

    return image

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

if __name__ == '__main__':

    # Read in the image
    data = fits.getdata('nnid0k3wiwq_blv_tmp.fits', 1)

    # Get subset of image to test with
    data = data[0:1000, 0:1000]

    # Save the test image
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(data, cmap='gray', vmin=-20, vmax=20)
    plt.savefig('test_data.png')
    if os.path.exists('test.fits'):
        os.remove('test.fits')
    hdu = fits.PrimaryHDU()
    hdu1 = fits.ImageHDU(data)
    hdulist = fits.HDUList([hdu,hdu1])
    hdulist.writeto('test.fits')

    # # Make histogram to find best threshold
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.hist(data, range=(-20,20), bins=100)
    # plt.savefig('hist.png')

    # Binary threshold the image
    threshold = 20
    data[np.where(data < threshold)] = 0
    data[np.where(data >= threshold)] = 1

    # Save the binary thresholded image
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(data, cmap='gray')
    plt.savefig('binary.png')
    if os.path.exists('binary.fits'):
        os.remove('binary.fits')
    hdu = fits.PrimaryHDU()
    hdu1 = fits.ImageHDU(data)
    hdulist = fits.HDUList([hdu,hdu1])
    hdulist.writeto('binary.fits')

    # Mark the cosmic rays
    mark_image(data)

    # Save the marked image
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(data, cmap='gray')
    plt.savefig('marked.png')
    if os.path.exists('marked.fits'):
        os.remove('marked.fits')
    hdu = fits.PrimaryHDU()
    hdu1 = fits.ImageHDU(data)
    hdulist = fits.HDUList([hdu,hdu1])
    hdulist.writeto('marked.fits')

    # Perform statistics

    # Set marked pixels back to 1
    data[np.where(data > 0)] = 1

    # Perform dilation (twice)
    dilated_image = dilate_image(data)
    dilated_image = dilate_image(dilated_image)

    # Save the dilated image
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(dilated_image, cmap='gray')
    plt.savefig('dilated.png')
    if os.path.exists('dilated.fits'):
        os.remove('dilated.fits')
    hdu = fits.PrimaryHDU()
    hdu1 = fits.ImageHDU(dilated_image)
    hdulist = fits.HDUList([hdu,hdu1])
    hdulist.writeto('dilated.fits')
