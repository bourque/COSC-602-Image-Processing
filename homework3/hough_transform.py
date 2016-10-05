import numpy as np
from skimage.io import imread
from skimage.io import imsave

# -----------------------------------------------------------------------------

def hough(image_file, angles, pix_per_line, delta_theta=1.0, delta_p = 1.0, threshold=100):
    """

    """

    # Open the image file
    image = imread(image_file)

    # Initializations
    nrows = image.shape[0]
    ncols = image.shape[1]

    # Initialize transformed image
    rho = np.arange(0.0, (np.sqrt(2)*nrows)+1, delta_p)
    theta = np.arange(float(angles[0]), float(angles[1])+1, delta_theta)
    transformed_image = np.zeros((len(rho), len(theta))).astype(np.uint)

    # Perform the Hough transform
    for row in range(0,nrows):
        for col in range(0,ncols):
            if image[row,col] >= threshold:
                theta = angles[0]
                while(theta<=angles[1]):
                    p = np.abs(row*np.cos(theta*(np.pi/180.)) + col*np.sin(theta*(np.pi/180.)))
                    p = round(p, 1)
                    transformed_image[p, theta] += 1
                    theta += delta_theta

    # Threshold the resulting image for number of pixels per line:
    transformed_image[np.where(transformed_image >= pix_per_line)] = 255
    transformed_image[np.where(transformed_image < pix_per_line)] = 0

    # Save the image
    imsave(image_file.replace('.bmp', '_hough.bmp'), transformed_image)
    print 'Image saved to {}'.format(image_file.replace('.bmp', '_hough.bmp'))

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

if __name__ == '__main__':

    # Get the angles of interest from user
    prompt='\nPlease enter the line angle(s) of interest (e.g. "0-180"):\n'
    angle_of_interest = raw_input(prompt)
    assert '-' in angle_of_interest, 'Not a valid range. Please try again.'
    angle_of_interest = (float(angle_of_interest.split('-')[0]), float(angle_of_interest.split('-')[1]))
    assert 0 <= angle_of_interest[0] < 180, 'Not a valid range. Please try again.'
    assert 0 < angle_of_interest[1] <= 180, 'Not a valid range. Please try again.'

    # Get the minimum number of pixels per line from user
    prompt='\nPlease enter the minimum number of pixels per line (e.g. "10"):\n'
    pix_per_line = int(raw_input(prompt))

    # Perform the transform
    hough('cam_sobel.bmp', angle_of_interest, pix_per_line)