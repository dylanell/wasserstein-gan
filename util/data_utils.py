"""
General data handling utilities.
"""

import numpy as np
import imageio
import glob

def make_gif_from_numbered_images(wildcard_str, dest_path='/tmp/nice.gif'):
    """
    Description: Given a wildcard string to match all image files within a directory (e.g.
        '/tmp/mymodel*.png'), construct a gif from all files matching this wildcard. Requires
        image file names end with an order number (e.g. 'image_<order_number>.png') and gif is constructed in ascending order by order number. Saves gif to /tmp directory by default.
    Args:
        - wildcard_str (string): wildcard string to match all image files to make gif.
        - dest_path (string): path and filename to save gif.
    Returns:
        - None
    """

    # get all images that match wildcard string
    img_files = glob.glob(wildcard_str)

    # sort filenames by numbers
    img_files.sort(key=lambda f: int(f.split('_')[-1].split('.')[0]))

    # create list if image objects
    img_list = [imageio.imread(img) for img in img_files]

    # write image list to gif
    imageio.mimwrite(dest_path, img_list, fps=200)

def tile_images(imgs):
    """
    Description: Given a batch of images (must be a perfect square number of samples e.g. 64),
        organize them into a larger tiled image.
    Args:
        - image (4D numpy array): batch of 3D images.
    Returns:
        - tile_imgs (3D numpy array): single image of tiled image batch.
    """

    # scale pixel values to [0, 255] and cast to unsigned 8-bit integers (common image datatype)
    min_data, max_data = [float(np.min(imgs)), float(np.max(imgs))]
    min_scale, max_scale = [0., 255.]
    imgs = ((max_scale - min_scale) * (imgs - min_data) / (max_data - min_data)) + min_scale
    imgs = imgs.astype(np.uint8)

    # tile images to larger image
    n_dim, h_dim, w_dim, d_dim = imgs.shape
    b_h, b_w = int(np.sqrt(n_dim)), int(np.sqrt(n_dim))
    tile_imgs = np.zeros((b_h*h_dim, b_w*w_dim, d_dim), dtype=np.uint8)
    for t_idx in range(n_dim):
        n_idx = w_dim * (t_idx % b_w)
        m_idx = h_dim * (t_idx // b_h)
        tile_imgs[n_idx:(n_idx+h_dim), m_idx:(m_idx+w_dim), :] = imgs[t_idx]

    return tile_imgs
