import numpy as np
from skimage.io import imread
from skimage.io import imsave
import matplotlib.pyplot as plt

def hough(image_file, angles, pix_per_line):
    """

    """

    # Open the image file
    image = imread(image_file)

    # Initializations
    nrows = image.shape[0]
    ncols = image.shape[1]
    delta_theta = 1.
    delta_p = 1.
    threshold = 100

    # Initialize transformed image
    rho = np.arange(0.0, (np.sqrt(2)*nrows)+1, delta_p)
    theta = np.arange(float(angles[0]), float(angles[1])+1, delta_theta)
    transformed_image = np.zeros((len(rho), len(theta))).astype(np.uint)

    # Perform the Hough transform
    for row in range(0,nrows):
        for col in range(0,ncols):
            if image[row,col] >= threshold:
                theta = 0.0
                while(theta<=180.0):
                    p = np.abs(row*np.cos(theta*(np.pi/180.)) + col*np.sin(theta*(np.pi/180.)))
                    p = round(p, 1)
                    transformed_image[p, theta] += 1
                    theta += delta_theta

    # Threshold the resulting image for number of pixels per line:
    transformed_image[np.where(transformed_image >= pix_per_line)] = 255
    transformed_image[np.where(transformed_image < pix_per_line)] = 0

    # Save the image
    imsave('cam_hough.bmp', transformed_image)

    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.imshow(transformed_image)
    # plt.show()


if __name__ == '__main__':

    hough('cam_sobel.bmp', (0,180), 10)