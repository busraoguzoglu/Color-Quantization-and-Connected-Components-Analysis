from PIL import Image
from matplotlib import pyplot as plt
import numpy as np
from math import sqrt


def main():

    im = Image.open("2.jpg")
    val1 = input("Enter k: ")
    k = int(val1)  # #number of clusters used in k means algo
    quantize(im, k)


def quantize(im, k):

    # Step 1: Convert input image to a matrix of pixels.

    image_sequence = im.getdata()
    image_array = np.array(image_sequence)
    image_array = np.float32(image_array.reshape(-1, 3))
    plt.imshow(im)

    # Click or Random?
    val2 = input("Enter 1 for selecting points manually, other value for random selection: ")
    print(val2)
    click = int(val2)

    width, height = im.size
    rgb_im = im.convert('RGB')

    # Step 2: Choose Points From image
    # Getting points from the image by clicking

    if click == 1:

        points_by_click = plt.ginput(k, show_clicks=True)
        print("clicked", points_by_click)
        # print(points_by_click[1])

        # Convert the clicked locations to color centers:
        color_centers = {}
        for i in range(k):
            color_centers[i] = (rgb_im.getpixel(points_by_click[i]))
        print(color_centers)

    # Get random color centers:
    if click != 1:

        color_centers = {}
        for i in range(k):
            rand_x = np.random.uniform(0, width)
            rand_y = np.random.uniform(0, height)
            color_centers[i] = (rgb_im.getpixel((rand_x, rand_y)))
        print(color_centers)

    # Step 3: K-Means Algorithm

    max_iteration = 5                   # Limiting the number of iterations to give a stop criteria

    clusters = {}

    for y in range(k):  # set k clusters
        clusters[y] = []
        clusters[y].append(color_centers[y])

    print('initial clusters are:', clusters)

    for iteration_count in range(max_iteration):

        x = 0

        for rgb_pixel in image_array:

            distances = [np.linalg.norm(rgb_pixel - (clusters[cluster])[0]) for cluster in clusters]
            classified_cluster = distances.index(min(distances))
            # convert this rgb pixel to list item
            r, g, b = rgb_pixel
            clusters[classified_cluster].append((r, g, b))  # Add this pixel to the cluster

        # clusters are created, find new mean value, change with 0th element.
        # Updating means
        for cluster in clusters:

            size_for_loop = len((clusters[cluster]))
            size_for_mean = len((clusters[cluster]))
            r, g, b = (clusters[cluster])[0]

            for i in range (size_for_loop-1):
                this_r, this_g, this_b = (clusters[cluster])[i+1]
                # Add all reds add all blues add all greens
                r += this_r
                g += this_g
                b += this_b

            r /= size_for_mean
            g /= size_for_mean
            b /= size_for_mean

            (clusters[cluster])[0] = (r, g, b)  # here, put mean of all values (new mean/center color)

        # Clean the clusters except mean/center (or create new clusters with only means)
        new_clusters = {}
        cluster_iteration = 0

        for cluster in clusters:  # set k clusters
            new_clusters[cluster_iteration] = []
            new_clusters[cluster_iteration].append((clusters[cluster])[0])
            cluster_iteration += 1
        clusters = new_clusters

        # Update iteration count, check all pixels again

        # Print final means:
        if iteration_count == max_iteration-1:
            final_cluster_means = clusters
            print('final cluster means are:', final_cluster_means)

    # Generate the output image (each pixel should have the color of assigned cluster)
    # Can use final_cluster_means
    # processed_im = Image.open("Ducky_Head.jpg")
    processed_im = im
    width, height = processed_im.size

    # Process every pixel
    for x in range(width):

        for y in range(height):
            current_color = processed_im.getpixel( (x,y) )

            assigned_mean = 0

            r_initial, g_initial, b_initial = current_color

            r, g, b = final_cluster_means[0][0]
            r = int(r)
            g = int(g)
            b = int(b)

            min_distance = sqrt((r_initial-r)**2+(g_initial-g)**2+(b_initial-b)**2)

            for i in range(k-1):

                r_new, g_new, b_new = final_cluster_means[i+1][0]
                r_new = int(r_new)
                g_new = int(g_new)
                b_new = int(b_new)

                new_distance = sqrt((r_initial-r_new)**2+(g_initial-g_new)**2+(b_initial-b_new)**2)

                if new_distance < min_distance:
                    min_distance = new_distance
                    assigned_mean = i+1

            # Assign the mean:
            r, g, b = final_cluster_means[assigned_mean][0]
            r = int(r)
            g = int(g)
            b = int(b)
            new_color = r, g, b

            processed_im.putpixel((x, y), new_color)

    plt.imshow(processed_im)
    plt.show()

if __name__ == '__main__':
    main()
