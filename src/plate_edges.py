import matplotlib.pyplot as plt
import numpy as np

from skimage import data, color, img_as_ubyte
from skimage.feature import canny
from skimage.transform import hough_ellipse, hough_circle_peaks, rescale, resize, hough_circle
from skimage.draw import ellipse_perimeter, circle_perimeter
from skimage import io
import os

def main():
    data_file = "data"
    for p in os.listdir(data_file):
        im_path = os.path.join(data_file, p)
        find_plate(im_path)

def find_plate(path):
    # Load picture, convert to grayscale and detect edges
    image_rgb = io.imread(path)
    image_rgb = rescale(image_rgb, 0.05)
    image_gray = color.rgb2gray(image_rgb)
    edges = canny(image_gray, sigma=2.0,
                  low_threshold=0.2, high_threshold=0.8)


    # Perform a Hough Transform
    # The accuracy corresponds to the bin size of a major axis.
    # The value is chosen in order to get a single high accumulator.
    # The threshold eliminates low accumulators
    result = hough_ellipse(edges)
    # plt.imshow(edges)
    # plt.show()
    # print (0)
    # result = hough_ellipse(edges)
    print(result.size)
    if result.size == 0:
        print(path + "\t non detected")
        return

    print(path + "\t detected")
    result.sort(order='accumulator')

    # Estimated parameters for the ellipse
    best = list(result[-1])
    yc, xc, a, b = [int(round(x)) for x in best[1:5]]
    orientation = best[5]

    # Draw the ellipse on the original image
    cy, cx = ellipse_perimeter(yc, xc, a, b, orientation)
    try:
        image_rgb[cy, cx] = (0, 0, 255)
        edges = color.gray2rgb(img_as_ubyte(edges))
        edges[cy, cx] = (250, 0, 0)
    except IndexError as e:
        print(e)
    # Draw the edge (white) and the resulting ellipse (red)


    fig2, (ax1, ax2) = plt.subplots(ncols=2, nrows=1, figsize=(8, 4),
                                    sharex=True, sharey=True)

    ax1.set_title('Original picture')
    ax1.imshow(image_rgb)

    ax2.set_title('Edge (white) and result (red)')
    ax2.imshow(edges)

    plt.show()

def find_plate2(path):
    # Load picture and detect edges
    image_rgb = io.imread(path)
    image = color.rgb2gray(image_rgb)
    # image = resize(image, (100, 100))
    image = rescale(image, 0.1)
    edges = canny(image, sigma=2, low_threshold=0.40, high_threshold=0.80)
    plt.imshow(edges)
    plt.show()
    # Detect two radii
    hough_radii = np.arange(20, 60, 2)
    hough_res = hough_circle(edges, hough_radii)

    # Select the most prominent 5 circles
    accums, cx, cy, radii = hough_circle_peaks(hough_res, hough_radii,
                                               total_num_peaks=3)

    # Draw them
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(10, 4))
    image = color.gray2rgb(image)
    for center_y, center_x, radius in zip(cy, cx, radii):
        circy, circx = circle_perimeter(center_y, center_x, radius)
        try:
            image[circy, circx] = (220, 20, 20)
        except IndexError as e:
            print (e)

    ax.imshow(image, cmap=plt.cm.gray)
    plt.show()

main()