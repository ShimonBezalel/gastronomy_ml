import numpy as np
import os
from skimage import io
import matplotlib.pyplot as plt
import math
import json
from pprint import pprint as pp

from skimage import data
from skimage.util.dtype import dtype_range
from skimage.util import img_as_ubyte
from skimage import exposure
from skimage.morphology import disk
from skimage.filters import rank
from skimage.transform import rescale, resize
from skimage.color import rgb2hsv

from skimage import data, color, img_as_ubyte
from skimage.feature import canny
from skimage.transform import hough_ellipse
from skimage.draw import ellipse_perimeter

from scipy.ndimage import gaussian_filter



def old():
    data_file = "data_negative/shittyfood"

    tuples = []
    i = 0
    t = []
    for p in os.listdir(data_file):
        t.append(p)
        i += 1
        if i % 3 == 0:
            tuples.append(t)
            t = []

    for im_path in os.listdir(data_file):

        fig = plt.figure(figsize=(8, 5))
        axes = np.zeros((2, 3), dtype=np.object)
        axes[0, 0] = plt.subplot(2, 3, 1)
        axes[0, 1] = plt.subplot(2, 3, 2, sharex=axes[0, 0], sharey=axes[0, 0])
        axes[0, 2] = plt.subplot(2, 3, 3, sharex=axes[0, 0], sharey=axes[0, 0])
        axes[1, 0] = plt.subplot(2, 3, 4)
        axes[1, 1] = plt.subplot(2, 3, 5)
        axes[1, 2] = plt.subplot(2, 3, 6)

        ims = []
        im = io.imread(os.path.join(data_file, im_path))
        small = resize(im, (100, 100, 3))
        hsv = rgb2hsv(small)

        for i in range(3):
            # im = io.imread(os.path.join(data_file, tup[i]))
            ims.append( hsv[..., i] )



        ax_img, ax_hist, ax_cdf = plot_img_and_hist(ims[0], axes[:, 0])
        ax_img.set_title('H')
        ax_hist.set_ylabel('Number of pixels')

        ax_img, ax_hist, ax_cdf = plot_img_and_hist(ims[1], axes[:, 1])
        ax_img.set_title('S')

        ax_img, ax_hist, ax_cdf = plot_img_and_hist(small, axes[:, 2])
        ax_img.set_title('O')
        ax_cdf.set_ylabel('Fraction of total intensity')

        # prevent overlap of y-axis labels
        fig.tight_layout()
        plt.show()


def proc_hist(src, index, i, query):
    print("retreiving filtered data from index")
    with open(index, 'r') as f:
        jindex = json.load(f)
    relevant = jindex["queries"][i]
    assert relevant["query"] == query
    relevant = set(relevant["images"])

    print("opening images {}...".format(len(relevant)))

    all_ims = [resize(io.imread(os.path.join(src, im_path)), (100, 100, 3))
               for im_path in filter(lambda i: i in relevant, sorted(os.listdir(src)))]
    print("opened {}.".format(len(all_ims)))
    bins = 10
    for standard in ['rgb', 'hsv']:
        print("processing for {}...".format(standard))
        all_hists = np.zeros((len(all_ims), 3, bins))

        for i, small in enumerate(all_ims):
            if standard == 'hsv':
                small = rgb2hsv(small)

            hists = np.zeros((3, bins)).astype(np.float16)
            for j in range(3):
                channel = small[..., j]
                hist = np.histogram(channel, bins=bins, range=(0, 1), density=False)
                hists[j] = hist[0]
            all_hists[i] = hists
            if i % 100 == 0:
                print("\t{}".format(str(i)))

        all_hists /= 10000
        np.save("processed/{}_hist_feature1_dish".format(standard), all_hists)

def closest_color_index(color, colors):
    dist = np.linalg.norm(colors - color, axis=1)
    return np.argmin(dist)

def proc_common_color_vec(src, dest, index, i, query):
    print("loading common colors...")

    common_colors = np.load("1000colors.npy")
    # plt.imshow(common_colors.reshape(100,10,3))
    # plt.show()
    sorted_colors = common_colors[np.argsort(common_colors, axis=0)[..., 0]]
    # plt.imshow(sorted_colors.reshape(100,10,3))
    # plt.show()
    # return

    print("retreiving filter from index")
    with open(index, 'r') as f:
        jindex = json.load(f)
    relevant = jindex["queries"][i]
    assert relevant["query"] == query
    relevant = set(relevant["images"])

    print("Retreiving data from {}".format(src))
    with open(src, 'r') as f:
        jdata = json.load(f)

    res = np.zeros((len(relevant), common_colors.shape[0]))

    for i, image in enumerate(filter(lambda im: im["name"] in relevant,
                                     sorted(jdata["images"], key=lambda k: k["name"]))):

        colors = image["features"]["imagePropertiesAnnotation"]["dominantColors"]["colors"]
        d = dominant_color_vector(colors, common_colors=sorted_colors, filter=False)
        res[i] = d

    np.save(dest, res)



def dominant_color_vector(colors, common_colors, filter=True, sigma_multiplyer=3):
    # A cubed matrix representing the colors with 3 dimensions, red, green and blue
    feature = np.zeros((common_colors.shape[0], )).astype(np.float64)
    for color in colors:
        color_mat = np.zeros_like(feature)
        color_obj = color["color"]
        color_vec = color_obj["red"], color_obj["green"], color_obj["blue"]
        score = color["score"]
        fraction = color["pixelFraction"]

        new_color_i = closest_color_index(np.array(color_vec) / 255, common_colors)

        # the binned values act as index for the result feature matrix
        # The fraction gives us an indicator how dominant the color was in the original image
        # color_mat[new_color_i] = fraction
        color_mat[new_color_i] = 1
        # the score will set a radius for influence on other colors.
        if filter:
            color_mat = gaussian_filter(color_mat,
                                        sigma=fraction * sigma_multiplyer,
                                        mode="constant")
        feature += color_mat

    return feature



def plot_img_and_hist(image, axes, bins=10):
    """Plot an image along with its histogram and cumulative histogram.

    """
    ax_img, ax_hist = axes
    ax_cdf = ax_hist.twinx()

    # Display image
    ax_img.imshow(image, cmap=plt.cm.gray)
    ax_img.set_axis_off()
    # Display histogram
    ax_hist.hist(image.ravel(), bins=bins)
    ax_hist.ticklabel_format(axis='y', style='scientific', scilimits=(0, 0))
    ax_hist.set_xlabel('Pixel intensity')

    xmin, xmax = dtype_range[image.dtype.type]
    # ax_hist.set_xlim(xmin, xmax)
    ax_hist.set_xlim(0, 1)

    # Display cumulative distribution
    img_cdf, bins = exposure.cumulative_distribution(image, bins)
    ax_cdf.plot(bins, img_cdf, 'r')

    return ax_img, ax_hist, ax_cdf

def gen_random_colors(num=1000, high_score=0.2):
    random_colors = {"colors": []}
    for i in range(num):
        r_color = np.random.randint(0, 255, size=3)
        obj = {
            "color": {
                "red": int(r_color[0]),
                "green": int(r_color[1]),
                "blue": int(r_color[2])
            },
            "score": np.random.uniform(0, high_score),
            "pixelFraction": np.random.uniform(0, 0.8)
        }
        random_colors["colors"].append(obj)
    with open("random_colors.json", 'w') as f:
        json.dump(random_colors, f)

def dominant_color_matrix(colors, bins=10, filter=True, sigma_multiplyer=3):
    # A cubed matrix representing the colors with 3 dimensions, red, green and blue
    feature = np.zeros((bins, bins, bins)).astype(np.float64)
    a_bins = np.arange(0, 255, 255/bins)
    for color in colors:
        color_mat = np.zeros((bins, bins, bins)).astype(np.float64)
        color_obj = color["color"]
        color_vec = color_obj["red"], color_obj["green"], color_obj["blue"]
        score = color["score"]
        fraction = color["pixelFraction"]
        bin_index = np.digitize(color_vec, a_bins)
        # binned = round(r / bins), round(g / bins), round(b / bins)

        # the binned values act as index for the result feature matrix
        # The fraction gives us an indicator how dominant the color was in the original image
        color_mat[tuple(bin_index - 1)] = fraction

        # the score will set a radius for influence on other colors.
        if filter:
            color_mat = gaussian_filter(color_mat,
                                        sigma=fraction * sigma_multiplyer,
                                        mode="constant")
        feature += color_mat

    return feature

def check():
    n = np.load("processed/color_feature3_nofilter.npy")
    print(np.max(n))

def proc_random_colors(num=1000):
    with open("random_colors.json", 'r') as f:
        r_colors = json.load(f)
    colors = r_colors["colors"]
    bins = 10
    res = np.zeros((num//10, bins, bins, bins))
    for i, j in enumerate(range(0, num, 10)):
        seg = colors[j:j+10]

        d = dominant_color_matrix(seg, bins=bins, filter=False)
        res[i] = d
    np.save("random_colors_processed", res)

def visualize_colors(colors, width=1000, height=10, sort_colors=False):
    strip = np.zeros((height, width, 3)).astype(np.float64)
    # a_bins = np.arange(0, 255, 255/bins)
    sum_of_frac = 0
    if sort_colors:
        colors = sorted(colors, key=lambda c: c["pixelFraction"])
    for color in colors:
        fraction = color["pixelFraction"]
        sum_of_frac += fraction

    pos = 0
    for color in colors:
        color_obj = color["color"]
        color_vec = color_obj["red"], color_obj["green"], color_obj["blue"]
        # score = color["score"]
        fraction = color["pixelFraction"]
        normal_frac = fraction / sum_of_frac


        # the binned values act as index for the result feature matrix
        # The fraction gives us an indicator how dominant the color was in the original image
        strip[::, int(pos*width):int((pos+normal_frac)*width)] = color_vec

        pos += normal_frac


    return strip/255

def visualize_random():
    path = "/Users/shimonheimowitz/PycharmProjects/michelin_star/random_colors.json"

    with open(path, 'r') as f:
        jcolors = json.load(f)

    all_colors = jcolors["colors"]
    height = 10
    width = 1000
    im = np.zeros((height * 500, width, 3))
    im = np.zeros((height * 500, width, 3))

    color_pos = 0
    height_pos = 0
    while color_pos < len(all_colors):
        size = np.random.randint(low=7, high=10)
        colors = all_colors[color_pos:color_pos+size]
        strip = visualize_colors(colors, width=width, height=height, sort_colors=True)
        im[height_pos:height_pos + height] = strip
        height_pos += height
        color_pos += size
        if height_pos // 10 >= 489:
            break
    plt.imshow(im)
    plt.imsave("dominant_colors_random_sorted.jpg", im)
    plt.show()


def visualize(path, index, query="Dish", i=0):
    with open(path, 'r') as f:
        jdata = json.load(f)
    with open(index, 'r') as f:
        jindex = json.load(f)
    relevant = jindex["queries"][i]
    assert relevant["query"] == query
    relevant = set(relevant["images"])
    height = 1
    width = 500
    im = np.zeros((height * len(relevant), width, 3))
    pos = 0
    for i, image in enumerate(filter(lambda im: im["name"] in relevant,
                                     sorted(jdata["images"], key=lambda k: k["name"]))):

        colors = image["features"]["imagePropertiesAnnotation"]["dominantColors"]["colors"]
        # d = dominant_color_matrix(colors, bins=bins, filter=False)
        strip = visualize_colors(colors, width=width, height=height, sort_colors=True)
        im[pos:pos+height] = strip
        pos += height
        if i == 15000:
            break

    plt.imshow(im)
    plt.imsave("all_colors.jpg", im)
    plt.show()

def visualize_all(paths):
    height = 1
    width = 500
    im = np.zeros((height * 10000, width, 3))
    pos = 0
    for path in paths:
        with open(path, 'r') as f:
            jdata = json.load(f)


        for i, image in enumerate(sorted(jdata["images"], key=lambda k: k["name"])):

            colors = image["features"]["imagePropertiesAnnotation"]["dominantColors"]["colors"]
            # d = dominant_color_matrix(colors, bins=bins, filter=False)
            strip = visualize_colors(colors, width=width, height=height, sort_colors=True)
            im[pos:pos+height] = strip
            pos += height
            if pos == 9999:
                break

    plt.imshow(im)
    plt.imsave("all_colors.jpg", im)
    plt.show()

def run(path, index, query, i=0, store_color=None):
    with open(path, 'r') as f:
        jdata = json.load(f)
    if store_color is None:
        # color_feature5_neg_plate_nofilter
        store_color = "processed/default" + str(np.random.randint(0, 10000))
    with open(index, 'r') as f:
        jindex = json.load(f)
    relevant = jindex["queries"][i]
    assert relevant["query"] ==  query
    relevant = set(relevant["images"])
    # order matters, so always sort image names
    bins=10
    res = np.zeros((len(relevant), bins, bins, bins))

    for i, image in enumerate(filter(lambda im: im["name"] in relevant,
                                     sorted(jdata["images"], key=lambda k: k["name"]))):

        colors = image["features"]["imagePropertiesAnnotation"]["dominantColors"]["colors"]
        d = dominant_color_matrix(colors, bins=bins, filter=False)
        res[i] = d

    np.save(store_color, res)

def main():
    for index, path in [("index_neg_med.json",  "foodphotography_images.json")]:
            # [("index_neg.json",      "shittyfood_images.json"),
            #             ("index_neg_med.json",  "foodphotography_images.json"),
            #             ("index.json",          "michelin_images.json")]:
        for query in ["Dish", "Plate"]:
            i = 2 if 'neg' not in index else 1
            i = 0 if query == "Plate" else i
            label = 'mic' if 'neg' not in index else 'neg'
            label += "med" if 'med' in index else ""
            dest = "processed/dominant_color_vec_feature{}_{}_{}".format(2, label,
                                                                         query.lower())
            proc_common_color_vec(src=path,
                                  dest=dest,
                                  index=index,
                                  query=query,
                                  i=i)

    # run(path, index, query=query, i=0, store_color="processed/color_feature6_neg_plate_nofilter")
    # visualize_all(["michelin_images.json", "shittyfood_images.json", "foodphotography_images.json"])
    # old()


main()



