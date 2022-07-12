import os
import pdb
import shutil
import tempfile

import imageio
import matplotlib.pyplot as plt
import numpy as np


def plt_formal_to_image(**kwargs):
    path = tempfile.mktemp(".png")
    plt_formal_save(path, **kwargs)
    img = imageio.imread(path)
    os.remove(path)
    plt.close()
    return img


def plt_formal_save(path, axis=False, legend=False, title=None):
    if not axis:
        plt.gca().set_axis_off()
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
    else:
        plt.grid()
    if legend:
        plt.legend()
    if title:
        plt.title(title)
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.savefig(path, bbox_inches="tight", pad_inches=0, dpi=300)
