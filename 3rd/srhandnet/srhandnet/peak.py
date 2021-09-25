import torch
import torch.nn as nn


def peak_local_max(img,
                   threshold,
                   kernel_size=3):
    """Return Local Maximum Pixel

    Parameters
    ----------
    img : torch.Tensor
        input image
    threshold : float
        threshold of intensities
    kernel_size : int
        size of local kernel

    Returns
    -------
    intensities : torch.Tensor
        local maximum value.
    coord : torch.Tensor
        vu coords.
    """
    mp = nn.MaxPool2d(kernel_size=kernel_size,
                      padding=(kernel_size - 1) // 2,
                      stride=1)
    maximum_img = mp(img)
    mask = img == maximum_img

    coord = torch.nonzero(mask)
    intensities = img[mask]
    indices = intensities > threshold
    return intensities[indices], coord[indices]
