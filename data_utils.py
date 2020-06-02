"""
General data handling utilities.
"""

import numpy as np

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
