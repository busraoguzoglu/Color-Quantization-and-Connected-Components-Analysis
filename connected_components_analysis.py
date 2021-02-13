import cv2 as cv
import numpy as np
import sys
import matplotlib.pyplot as plt


def main():

    im = cv.imread('birds1.jpg')
    threshold = 120
    object_count = countConnectedComponents(im, threshold)

    print('object count is:', object_count)


def countConnectedComponents(im, threshold):

    # Part 1: Read the image and apply a thresholding function:
    im = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
    ret, binary_im = cv.threshold(im, threshold, 255, cv.THRESH_BINARY_INV)

    plt.subplot(1, 3, 1)
    plt.imshow(im, cmap="gray")

    plt.subplot(1, 3, 2)
    plt.imshow(binary_im, cmap="gray")

    # Part 2: Clean the image
    # When these operations used in different orders on different images,
    # Process finished with exit code 139 (interrupted by signal 11: SIGSEGV)
    # sometimes appears in PNG/dice pictures. Problem in Ubuntu? Possibly

    binary_im = cv.medianBlur(binary_im, 3)
    kernel = np.ones((5, 5), np.uint8)

    #binary_im = cv.morphologyEx(binary_im, cv.MORPH_CLOSE, kernel)
    #binary_im = cv.morphologyEx(binary_im, cv.MORPH_DILATE, kernel)  # Gives better count for some bird pictures, bad for dice
    #binary_im = cv.morphologyEx(binary_im, cv.MORPH_ERODE, kernel)  # Used this in dice pictures
    #binary_im = cv.morphologyEx(binary_im, cv.MORPH_OPEN, np.ones([5, 5]))

    # Part 3: Connected Components analysis

    label_count = 0
    size = binary_im.shape
    width = size[0]
    height = size[1]

    # Utility function to create a list of white points on the binary picture created

    def create_white_list():
        white_list = []
        for x in range(width):
            for y in range(height):
                current_color = binary_im[x][y]
                if current_color == 255:
                    white_list.append((x, y))

        return white_list

    white_point_list = create_white_list()

    sys.setrecursionlimit(100000)  # To avoid getting stack overflow

    def recursive_labeling(pixel, im, label):

        im[pixel[0]][pixel[1]] = label

        # Check around 8 pixel to find possible connected white spot

        # Case 1: Upper left
        x = pixel[0] - 1
        y = pixel[1] + 1

        check1 = 1
        if y < 0 or y >= height-1 or x < 0 or x >= width-1:
            check1 = 0

        if check1 and im[x][y] == 255:
            recursive_labeling((x, y), im, label)

        # Case 2: Up
        x = pixel[0]
        y = pixel[1] + 1

        check2 = 1
        if y < 0 or y >= height-1 or x < 0 or x >= width-1:
            check2 = 0

        if check2 and im[x][y] == 255:
            recursive_labeling((x, y), im, label)

        # Case 3: Upper right
        x = pixel[0] + 1
        y = pixel[1] + 1

        check3 = 1
        if y < 0 or y >= height-1 or x < 0 or x >= width-1:
            check3 = 0

        if check3 and im[x][y] == 255:
            recursive_labeling((x, y), im, label)

        # Case 4: Left
        x = pixel[0] - 1
        y = pixel[1]

        check4 = 1
        if y < 0 or y >= height-1 or x < 0 or x >= width-1:
            check4 = 0

        if check4 and im[x][y] == 255:
            recursive_labeling((x, y), im, label)

        # Case 5: Right
        x = pixel[0] + 1
        y = pixel[1]

        check5 = 1
        if y < 0 or y >= height-1 or x < 0 or x >= width-1:
            check5 = 0

        if check5 and im[x][y] == 255:
            recursive_labeling((x, y), im, label)

        # Case 6: Down left
        x = pixel[0] - 1
        y = pixel[1] - 1

        check6 = 1
        if y < 0 or y >= height-1 or x < 0 or x >= width-1:
            check6 = 0

        if check6 and im[x][y] == 255:
            recursive_labeling((x, y), im, label)

        # Case 7: Down
        x = pixel[0]
        y = pixel[1] - 1

        check7 = 1
        if y < 0 or y >= height-1 or x < 0 or x >= width-1:
            check7 = 0

        if check7 and im[x][y] == 255:
            recursive_labeling((x, y), im, label)

        # Case 8: Down right
        x = pixel[0] + 1
        y = pixel[1] - 1

        check8 = 1
        if y < 0 or y >= height-1 or x < 0 or x >= width-1:
            check8 = 0

        if check8 and im[x][y] == 255:
            recursive_labeling((x, y), im, label)

    while len(white_point_list) > 0:
        recursive_labeling(white_point_list[0], binary_im, label_count)
        label_count += 1
        # Update white point list after labeling
        white_point_list = create_white_list()

    print('finished')

    plt.subplot(1, 3, 3)
    plt.imshow(binary_im, cmap="nipy_spectral")  # nipy-spectral gives colored image of labels 1,2,3..

    plt.show()
    return label_count


if __name__ == '__main__':
    main()
