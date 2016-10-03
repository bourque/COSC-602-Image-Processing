import numpy as np
from skimage.io import imread
from skimage.io import imsave

# -----------------------------------------------------------------------------

def apply_mask(filter_mask):
    """

    """

    mask = filter_mask[0]
    mask_name = filter_mask[1]
    print 'Applying {}'.format(mask_name)

    # Test image
    image = imread('bowl.jpg')
    image_with_noise = imread('bowl_with_noise.jpg')

    # Initializations
    modified_image = np.copy(image)
    modified_image_with_noise = np.copy(image_with_noise)
    nrows = image.shape[0] - 1
    ncols = image.shape[1] - 1

    # Iterate over the image
    for row in range(1,nrows):
        for col in range(1,ncols):

            # Gather the local neighborhood
            neighborhood = np.array([[image[row-1,col-1], image[row-1,col], image[row-1, col+1]],
                                     [image[row,col-1], image[row,col], image[row,col+1]],
                                     [image[row+1,col-1], image[row+1,col], image[row+1, col+1]]])
            neighborhood_with_noise = np.array([[image[row-1,col-1], image[row-1,col], image[row-1, col+1]],
                                                [image[row,col-1], image[row,col], image[row,col+1]],
                                                [image[row+1,col-1], image[row+1,col], image[row+1, col+1]]])

            # Convolve the neighborhood with the mask
            solution = np.sum(neighborhood * mask)
            solution_with_noise = np.sum(neighborhood_with_noise * mask)

            # Replace the central pixel with the convolution solution
            modified_image[row,col] = solution
            modified_image_with_noise[row,col] = solution_with_noise

    # Write out new image
    imsave('bowl_with_{}.jpg'.format(mask_name), modified_image)
    imsave('bowl_with_{}_with_noise.jpg'.format(mask_name), modified_image_with_noise)

# -----------------------------------------------------------------------------

def convert_user_mask_to_array(mask_elements):
    """

    """

    if len(mask_elements) == 9:
        mask = np.array([[mask_elements[0], mask_elements[1], mask_elements[2]],
                         [mask_elements[3], mask_elements[4], mask_elements[5]],
                         [mask_elements[6], mask_elements[7], mask_elements[8]]])

    return mask

# -----------------------------------------------------------------------------

if __name__ == '__main__':

    # Define the different types of masks
    mean_filter1 = np.array([[1/9., 1/9., 1/9.],
                             [1/9., 1/9., 1/9.],
                             [1/9., 1/9., 1/9.]])
    mean_filter2 = np.array([[1/10., 1/10., 1/10.],
                             [1/10., 2/10., 1/10.],
                             [1/10., 1/10., 1/10.]])
    mean_filter3 = np.array([[1/16., 2/16., 1/16.],
                             [2/16., 4/16., 2/16.],
                             [1/16., 2/16., 1/16.]])

    enhancement_filter1 = np.array([[-1., -1., -1.],
                                    [-1., 9., -1.],
                                    [-1., -1., -1.]])
    enhancement_filter2 = np.array([[1., -1., 1.],
                                    [-2., 5., -2.],
                                    [1., -2., 1.]])
    enhancement_filter3 = np.array([[0., -2., 0.],
                                    [-1., 5., -1.],
                                    [0., -1., 0.]])

    filter_dict = {}
    filter_dict['1'] = (mean_filter1, 'mean_filter_1')
    filter_dict['2'] = (mean_filter2, 'mean_filter_2')
    filter_dict['3'] = (mean_filter3, 'mean_filter_3')
    filter_dict['4'] = (enhancement_filter1, 'enhancement_filter_1')
    filter_dict['5'] = (enhancement_filter2, 'enhancement_filter_2')
    filter_dict['6'] = (enhancement_filter3, 'enhancement_filter_3')

    # Ask user what they want to do
    menu = '\nPlease select an option from the menu:\n'
    menu += '\t(1) Select pre-defined mask\n'
    menu += '\t(2) Provide a 3x3 mask\n'
    menu += '\t(3) Provide a 5x5 mask\n'
    response = raw_input(menu)
    assert response in ['1', '2', '3'], 'Not a valid response. Please try again.'

    # For pre-defined masks
    if response == '1':
        menu = '\nPlease select a mask:\n'
        menu += '\n(1) Mean filter 1: \n{}\n'.format(mean_filter1)
        menu += '\n(2) Mean filter 2: \n{}\n'.format(mean_filter2)
        menu += '\n(3) Mean filter 3: \n{}\n'.format(mean_filter3)
        menu += '\n(4) Enhancement filter 1: \n{}\n'.format(enhancement_filter1)
        menu += '\n(5) Enhancement filter 2: \n{}\n'.format(enhancement_filter2)
        menu += '\n(6) Enhancement filter 3: \n{}\n'.format(enhancement_filter3)
        response = raw_input(menu)
        assert response in ['1', '2', '3', '4', '5', '6'], 'Not a valid response.  Please try again.'
        apply_mask(filter_dict[response])

    # For user-suppled 3x3 mask
    if response == '2':
        prompt = '\nPlease enter 9 elements in order from left to right, top to bottom,\n'
        prompt += '\neach separated by a comma (e.g. "-1, -1, -1, -1, 9, -1, -1, -1, -1"):\n'
        mask_elements = raw_input(prompt)
        mask_elements = [float(item) for item in mask_elements.split(',')]
        mask = convert_user_mask_to_array(mask_elements)
        apply_mask((mask, 'user_3x3_mask'))
