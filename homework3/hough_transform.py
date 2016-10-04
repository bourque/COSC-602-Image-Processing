import numpy as np
from skimage.io import imread

if __name__ == '__main__':

    # Get image to test with
    image = imread('lines.jpg')

    # Initializations
    nrows = image.shape[0]
    ncols = image.shape[1]
    delta_theta = 0.1
    threshold = 255

    # Initialize transformed image
    rho = np.arange(0.0, np.sqrt(2)*nrows, 0.1)
    theta = np.arange(0.0, 180.0, delta_theta)
    transformed_image = np.zeros((len(rho), len(theta)))

    for row in range(0,nrows):
        for col in range(0,ncols):

            if image[row,col] >= threshold:

                theta = 0.0
                while(theta<=180.0):
                    p = np.abs(row*np.cos(theta) + col*np.sin(theta))
                    p = round(p, 1)
                    transformed_image[p, theta] += 1
                    theta += delta_theta