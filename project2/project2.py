import numpy as np
from astropy.io import fits

import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt

from skimage.transform import hough_line
from skimage.transform import hough_line_peaks
from skimage.transform import probabilistic_hough_line

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
# -----------------------------------------------------------------------------

if __name__ == '__main__':

    # Open up data and combine into one image
    hdulist = fits.open('icbh11n9q_flt.fits')
    nrows = hdulist[1].data.shape[0]
    ncols = hdulist[1].data.shape[1]
    data = np.zeros((nrows*2, ncols))
    data[0:nrows,:] = hdulist[1].data
    data[nrows:,:] = hdulist[4].data

    # Save copy of image
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111)
    ax.imshow(data, cmap='gray', vmin=0, vmax=180)
    plt.savefig('data.png')

    # Plot histogram
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111)
    ax.hist(data, range=(0,1000))
    plt.savefig('hist.png')

    # Binary threshold the data
    threshold = 80.0
    data_binary = np.copy(data)
    data_binary[np.where(data<=threshold)] = 0
    data_binary[np.where(data>threshold)] = 1

    # Save copy of thresholded image
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111)
    ax.imshow(data_binary, cmap='gray')
    plt.savefig('binary.png')

    hough, theta, dist = hough_line(data_binary)

    fig = plt.figure(figsize=(12,12))
    ax = fig.add_subplot(111)
    ax.plot(theta, dist, '+')
    plt.show()

    # # Initializations
    # vertical_mask, horizontal_mask = get_sobel_masks()
    # nrows = data_binary.shape[0] - 1
    # ncols = data_binary.shape[1] - 1
    # sobel_magnitude = np.copy(data_binary)
    # sobel_magnitude_with_roberts = np.copy(data_binary)

    # # Iterate over the image
    # print('\nCreating Sobel image:')
    # i=0
    # total = nrows * ncols
    # for row in range(1,nrows):
    #     for col in range(1,ncols):

    #         percent_completed = round(int((i/total)*100), 0)
    #         print('\t{}% Completed'.format(percent_completed), end='\r')
    #         i += 1

    #         neighborhood = get_neighborhood(data_binary, row, col)

    #         # Convolve the neighborhood with the Sobel mask
    #         vertical_solution = np.sum(neighborhood * vertical_mask)
    #         horizontal_solution = np.sum(neighborhood * vertical_mask)

    #         # Calculate the Sobel magnitude
    #         sobel_mag = np.sqrt(vertical_solution**2 + horizontal_solution**2)

    #         # Replace the pixel with the Sobel magnitude
    #         sobel_magnitude[row,col] = sobel_mag

    # # Remap the Sobel magnitude with Roberts
    # print('\nCreating Roberts image:')
    # i=0
    # for row in range(1,nrows):
    #     for col in range(1,ncols):

    #         percent_completed = round(int((i/total)*100), 0)
    #         print('\t{}% Completed'.format(percent_completed), end='\r')
    #         i += 1

    #         roberts = np.abs(sobel_magnitude[row,col] - sobel_magnitude[row-1,col-1]) + \
    #                   np.abs(sobel_magnitude[row,col-1] - sobel_magnitude[row-1,col])
    #         sobel_magnitude_with_roberts[row,col] = roberts

    # # Save copy of Sobel + Roberts image
    # fig = plt.figure(figsize=(12, 12))
    # ax = fig.add_subplot(111)
    # ax.imshow(sobel_magnitude_with_roberts, cmap='gray')
    # plt.savefig('sobel_with_roberts.png')
