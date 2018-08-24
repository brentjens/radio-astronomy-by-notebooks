"""Exploring the effect of partial sampling of the uv-plane

This module contains a simulation of how a radio interferometer "sees"
the sky by (only partially) observing the Fourier transform of the
sky. The module is used in the ``General-interferometry.ipynb`` notebook
for the 2018 Master course "Radio Astronomy" at Leiden University.

Please study the code carefully, and try to learn how to speed up
numerical calculations in Python, and how to properly document and
test your work to benefit other users of your code, as well as your
future self.

"""

from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt
import pyfftw
import pyfftw.interfaces.scipy_fftpack as fftwpack
import numba
import ipywidgets



'''
Example of good function documentation:
    """
    Solve the tensor equation ``a x = b`` for x.
    It is assumed that all indices of `x` are summed over in the product,
    together with the rightmost indices of `a`, as is done in, for example,
    ``tensordot(a, x, axes=b.ndim)``.
    Parameters
    ----------
    a : array_like
        Coefficient tensor, of shape ``b.shape + Q``. `Q`, a tuple, equals
        the shape of that sub-tensor of `a` consisting of the appropriate
        number of its rightmost indices, and must be such that
        ``prod(Q) == prod(b.shape)`` (in which sense `a` is said to be
        'square').
    b : array_like
        Right-hand tensor, which can be of any shape.
    axes : tuple of ints, optional
        Axes in `a` to reorder to the right, before inversion.
        If None (default), no reordering is done.
    Returns
    -------
    x : ndarray, shape Q
    Raises
    ------
    LinAlgError
        If `a` is singular or not 'square' (in the above sense).
    See Also
    --------
    numpy.tensordot, tensorinv, numpy.einsum
    Examples
    --------
    #>>> a = np.eye(2*3*4)
    #>>> a.shape = (2*3, 4, 2, 3, 4)
    #>>> b = np.random.randn(2*3, 4)
    #>>> x = np.linalg.tensorsolve(a, b)
    #>>> x.shape
    #(2, 3, 4)
    #>>> np.allclose(np.tensordot(a, x, axes=3), b)
    #True
    """
'''

@numba.njit('f4[:,:](f4[:,:])', parallel=True)
def dB(x):
    """
    Converts an array of positive, 32 bit floats to dB
    (10*log10(x)).

    Uses ``numba.njit`` to parallelize the code to keep
    it fast enough for interactive use. This function is meant to be
    used as the `scale` argument to the ``replot()`` function.

    Parameters
    ----------

    x : 2D ndarray of float32
        The numbers to convert.

    Returns
    -------
    2D ndarray of float32

    Examples
    --------
    >>> a = np.array([[0.1, 5, 100.0], [3, 0.01, 500]], dtype=np.float32)
    >>> dB(a)
    array([[-1.        ,  0.69897   ,  2.        ],
           [ 0.47712123, -2.        ,  2.69897   ]], dtype=float32)
    """
    nrows, ncols = x.shape
    result: numba.float32[:, :] = np.empty(x.shape, dtype=np.float32)
    for row in numba.prange(nrows):
        for col in range(ncols):
            result[row, col] = np.log10(x[row, col])
    return result




def linear(x):
    """
    Identity function, can be used as `scale` argument to the
    ``replot()`` function.

    Parameters
    ----------
    x : any Python object

    Returns
    -------
    x : the same instance

    Examples
    --------
    >>> a = []
    >>> a is linear(a)
    True
    """
    return x



@numba.njit('f4[:,:](i8, i8, f8, f8)')
def min_max_bl_mask(num_y, num_x, min_bl, max_bl):
    """
    Calculate a spatial fourier filter based on a minimum and maximum
    baseline  fraction, where  the maximum fraction == 1 corresponds
    to  `np.sqrt(mid_x**2 + mid_y**2)`

    Parameters
    ----------
    num_y  : int
             Number of rows in the image or corresponding uv-plane
    num_x  : int
             Number of columns in the image or corresponding uv-plane
    min_bl : float (0.0 <= x <= 1.0)
             Minimum baseline in terms of a fraction of the maximum
             possible baseline (baseline in one of the corners of the
             uv-plane)
    max_bl : float (0.0 <= x <= 1.0)
             Maximum baseline in terms of a fraction of the maximum
             possible baseline (baseline in one of the corners of the
             uv-plane)
    
    Returns
    -------
    Two-dimensional ``numpy.ndarray`` of ``float32`` of shape (num_y,
    num_x) containing the visibility weights.

    Examples
    --------
    >>> min_max_bl_mask(16, 14, 0.0, 1.0)
    array([[1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
           [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
           [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
           [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
           [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
           [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
           [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
           [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
           [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
           [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
           [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
           [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
           [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
           [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
           [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
           [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.]],
          dtype=float32)
    >>> min_max_bl_mask(16, 14, 0.4, 0.7)
    array([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0., 1., 1., 1., 1., 1., 0., 0., 0., 0.],
           [0., 0., 0., 1., 1., 1., 1., 1., 1., 1., 1., 1., 0., 0.],
           [0., 0., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 0.],
           [0., 1., 1., 1., 1., 1., 0., 0., 0., 1., 1., 1., 1., 1.],
           [0., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 1., 1., 1.],
           [1., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 1., 1., 1.],
           [1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1.],
           [1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1.],
           [1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1.],
           [1., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 1., 1., 1.],
           [0., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 1., 1., 1.],
           [0., 1., 1., 1., 1., 1., 0., 0., 0., 1., 1., 1., 1., 1.],
           [0., 0., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 0.],
           [0., 0., 0., 1., 1., 1., 1., 1., 1., 1., 1., 1., 0., 0.],
           [0., 0., 0., 0., 0., 1., 1., 1., 1., 1., 0., 0., 0., 0.]],
          dtype=float32)
    """
    mask: numba.float32[:, :] = np.empty((num_y, num_x), dtype=np.float32)
    mid_y, mid_x = num_y//2, num_x//2
    max_r = np.sqrt(mid_x**2 + mid_y**2)
    minf2 = (min_bl*max_r)**2
    maxf2 = (max_bl*max_r)**2
    for y in numba.prange(num_y):
        for x in range(num_x):
            r2 = (x-mid_x)**2 + (y - mid_y)**2
            if r2 >= minf2 and r2 <= maxf2:
                mask[y, x] = 1.0
            else:
                mask[y, x] = 0.0
    return mask





@dataclass
class SimulationOutput:
    image: np.ndarray
    uv_plane: np.ndarray
    weight: np.ndarray
    weighted_uv_plane: np.ndarray
    dirty_image: np.ndarray


def simulate(image, weights):
    '''
    image : 2D numpy.array of floats

    mask : 2D numpy.array of floats
    '''
    w = weights
    uv_plane = fftwpack.fftshift(fftwpack.ifft2(image, threads=4))
    weighted_uv_plane = pyfftw.empty_aligned(w.shape, dtype='complex64')
    weighted_uv_plane[:] = w*uv_plane
    dirty_image = fftwpack.fft2(fftwpack.fftshift(weighted_uv_plane), threads=4).real
    return SimulationOutput(image, uv_plane,
                            weights, weighted_uv_plane,
                            dirty_image)

pyfftw.interfaces.cache.enable()
pyfftw.interfaces.cache.set_keepalive_time(keepalive_time=100.0)

img_bw = pyfftw.empty_aligned((1024, 1024), dtype='float32')
img_bw[:] = np.array(plt.imread('shirt-bw.png')[:, :, :3].mean(axis=2),
                     dtype=np.float32)



fig = plt.figure()#(figsize=(8,8),dpi=180)
((ax_tl, ax_tr), (ax_bl, ax_br)) = fig.subplots(2, 2)
fig.subplots_adjust(hspace=0, wspace=0)
cbar_o = None
cbar_m = None
first_time = True

ax_tl_img = None
ax_tr_img = None
ax_bl_img = None
ax_br_img = None

previous_simresult = None

def replot(scale, img, min_bl_fraction, max_bl_fraction):
    global cbar_o, cbar_m
    global ax_tl_img, ax_tr_img, ax_bl_img, ax_br_img
    global previous_simresult
    mask = min_max_bl_mask(*(img[0].shape),
                           min_bl_fraction/100.0,
                           max_bl_fraction/100.0)
    sim_result = simulate(img[0], mask)
    # Original
    vmin, vmax = np.percentile(sim_result.image[::20, ::20], [1, 99])
    if ax_tl_img is None:
        ax_tl_img = ax_tl.imshow(sim_result.image,
                                 cmap=plt.cm.Greys_r,
                                 vmin=vmin, vmax=vmax,
                                 interpolation='nearest')
        ax_tl.set_xticks([])
        ax_tl.set_yticks([])
    else:
        ax_tl_img.set_data(sim_result.image)
        ax_tl_img.set_clim(vmin, vmax)

    # Original uv plane
    to_plot = scale(np.absolute(sim_result.uv_plane))
    vmin, vmax = np.percentile(to_plot[::20, ::20], (0.5, 99.0))
    if ax_tr_img is None:
        ax_tr_img = ax_tr.imshow(to_plot, vmax=vmax, vmin=vmin,
                                 interpolation='nearest')
        ax_tr.set_xticks([])
        ax_tr.set_yticks([])
    else:
        ax_tr_img.set_data(to_plot)
        ax_tr_img.set_clim(vmin, vmax)

    # Masked uv plane
    to_plot = scale(np.absolute(sim_result.weighted_uv_plane))
    if ax_br_img is None:
        ax_br_img = ax_br.imshow(to_plot, vmax=vmax, vmin=vmin,
                                 interpolation='nearest')
        ax_br.set_xticks([])
        ax_br.set_yticks([])
    else:
        ax_br_img.set_data(to_plot)
        ax_br_img.set_clim(vmin, vmax)

    # Dirty image
    vmin, vmax = np.percentile(sim_result.dirty_image[::20, ::20], [1, 99])
    if ax_bl_img is None:
        ax_bl_img = ax_bl.imshow(sim_result.dirty_image,
                                 cmap=plt.cm.Greys_r,
                                 vmin=vmin, vmax=vmax,
                                 interpolation='nearest')
        ax_bl.set_xticks([])
        ax_bl.set_yticks([])
    else:
        ax_bl_img.set_data(sim_result.dirty_image)
        ax_bl_img.set_clim(vmin, vmax)
    previous_simresult = sim_result
    fig.canvas.draw_idle() # Apparently a bit faster than fig.canvas.draw()


replot(linear, (img_bw,), 30, 80)

fig.show()

ipywidgets.interact(replot,
                    scale={'Linear': linear, 'dB': dB},
                    img={'Shirt': (img_bw,)},
                    min_bl_fraction=ipywidgets.FloatSlider(
                        min=0, max=100, step=0.25, value=0, continuous_update=True),
                    max_bl_fraction=ipywidgets.FloatSlider(
                        min=0, max=100, step=0.25, value=100, continuous_update=True))
