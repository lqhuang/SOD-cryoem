from __future__ import print_function, division

import os
import pickle
import argparse
from random import randint 
from matplotlib import pyplot as plt

from cryoio import mrc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_imgs', type=str)
    parser.add_argument('-r', '--random', type=bool, default=False)

    args = parser.parse_args()
    input_imgs = os.path.abspath(args.input_imgs)
    plot_randomly = args.random
    print(plot_randomly)
    image_stack = mrc.readMRCimgs(input_imgs, 0)
    size = image_stack.shape
    print('image size: {0}x{1}, number of images: {2}'.format(*size))
    plt.figure(1)
    for i in range(9):
        plt.subplot(331+i)
        if plot_randomly:
            num = randint(0, size[2])
        else:
            num = i
        print(num)
        plt.imshow(image_stack[:, :, num])
    plt.show()


if __name__ == '__main__':
    main()
