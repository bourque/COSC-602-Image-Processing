#! /usr/bin/env python

"""Apply a fast fourier transform and an inverste fast fourier
transform to a test image.

Author:
    Matthew Bourque, November, 2016

Use:
    The user can exectute this program from the command line as such:

        >>> python fourier_transform.py

Outputs:
    test_image.png - The image that is tested with the program.
    fft.png - The fast fourier transform of the test image.
    ifft.png - The inverse fast fourier transform of the fft produced.

Dependencies:
    The user must have a Python 3.5 installion.  The matplotlib, numpy,
    and scipy external libraries are also required.
"""

import matplotlib.pyplot as plt
import numpy as np
import scipy.fftpack

# -----------------------------------------------------------------------------

def ifft(image):
    """Perform an inverse fast fourier transform on the given image.

    Parameters
    ----------
    image : 2D array
        A 2D array of values to compute the inverse fast fourier
        transform on.

    Returns
    -------
    ift : 2D array
        A 2D array of the resulting inverse fast fourier transform.
    """

    ift = scipy.fftpack.ifftn(image)
    ift = np.real(ift)

    return ift

# -----------------------------------------------------------------------------

def fft(image):
    """Perform a fast fourier transform on the given image.

    Parameters
    ----------
    image : 2D array
        A 2D array of values to compute the fast fourier transform on.

    Returns
    -------
    ft : 2D array
        A 2D array of the resulting fast fourier transform.
    """

    ft = scipy.fftpack.fftn(image)
    ft = np.real(ft)

    return ft

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

if __name__ == '__main__':

    # Create image to test with
    image = np.zeros((50,50))
    image[25,25] = 1

    # Save the test image
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(image, cmap='gray', interpolation='none')
    plt.savefig('test_image.png')

    # Perform a fast fourier transform on the image
    image_fft = fft(image)

    # Perform the inverse fast fourier transform on the resulting fft
    image_ifft = ifft(image_fft)

    # Save the result of the fft
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(image_fft, cmap='gray')
    plt.savefig('fft.png')

    # Save the result of the ifft
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(image_ifft, cmap='gray', interpolation='none')
    plt.savefig('ifft.png')
