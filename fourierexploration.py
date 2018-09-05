"""Exploring the effect of partial sampling of the uv-plane

This module contains a simulation of how a radio interferometer "sees"
the sky by (only partially) observing the Fourier transform of the
sky. The module is used in the ``General-interferometry.ipynb`` notebook
for the 2018 Master course "Radio Astronomy" at Leiden University.

Please study the code carefully, and try to learn how to speed up
numerical calculations in Python, and how to properly document and
test your work to benefit other users of your code, as well as your
future self.

Good function documentation begins with a synopsis of the function's
behaviour and intended use, followed by sections describing its
parameters, its return values, potential exceptions raised, and
examples. The examples can be used by the ``pytest`` commandline
utility to perform elementary tests of the functions.

"""

from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcol
import pyfftw
import pyfftw.interfaces.scipy_fftpack as fftwpack
import numba
import ipywidgets



# '''
# Example of good function documentation:
#     """
#     Solve the tensor equation ``a x = b`` for x.
#     It is assumed that all indices of `x` are summed over in the product,
#     together with the rightmost indices of `a`, as is done in, for example,
#     ``tensordot(a, x, axes=b.ndim)``.
#     Parameters
#     ----------
#     a : array_like
#         Coefficient tensor, of shape ``b.shape + Q``. `Q`, a tuple, equals
#         the shape of that sub-tensor of `a` consisting of the appropriate
#         number of its rightmost indices, and must be such that
#         ``prod(Q) == prod(b.shape)`` (in which sense `a` is said to be
#         'square').
#     b : array_like
#         Right-hand tensor, which can be of any shape.
#     axes : tuple of ints, optional
#         Axes in `a` to reorder to the right, before inversion.
#         If None (default), no reordering is done.
#     Returns
#     -------
#     x : ndarray, shape Q
#     Raises
#     ------
#     LinAlgError
#         If `a` is singular or not 'square' (in the above sense).
#     See Also
#     --------
#     numpy.tensordot, tensorinv, numpy.einsum
#     Examples
#     --------
#     #>>> a = np.eye(2*3*4)
#     #>>> a.shape = (2*3, 4, 2, 3, 4)
#     #>>> b = np.random.randn(2*3, 4)
#     #>>> x = np.linalg.tensorsolve(a, b)
#     #>>> x.shape
#     #(2, 3, 4)
#     #>>> np.allclose(np.tensordot(a, x, axes=3), b)
#     #True
#     """
# '''

# @numba.njit('f4[:,:](f4[:,:])', parallel=True)
# def dB(x):
#     """
#     Converts an array of positive, 32 bit floats to dB
#     (10*log10(x)).

#     Uses ``numba.njit`` to parallelize the code to keep
#     it fast enough for interactive use. This function is meant to be
#     used as the `scale` argument to the ``replot()`` function.

#     Parameters
#     ----------

#     x : 2D ndarray of float32
#         The numbers to convert.

#     Returns
#     -------
#     2D ndarray of float32

#     Examples
#     --------
#     >>> a = np.array([[0.1, 5, 100.0], [3, 0.01, 500]], dtype=np.float32)
#     >>> dB(a)
#     array([[-1.        ,  0.69897   ,  2.        ],
#            [ 0.47712123, -2.        ,  2.69897   ]], dtype=float32)
#     """
#     nrows, ncols = x.shape
#     result: numba.float32[:, :] = np.empty(x.shape, dtype=np.float32)
#     for row in numba.prange(nrows):
#         for col in range(ncols):
#             result[row, col] = np.log10(x[row, col])
#     return result




# def linear(x):
#     """
#     Identity function, can be used as `scale` argument to the
#     ``replot()`` function.

#     Parameters
#     ----------
#     x : any Python object

#     Returns
#     -------
#     x : the same instance

#     Examples
#     --------
#     >>> a = []
#     >>> a is linear(a)
#     True
#     >>> b = 'bbb'
#     >>> 'bbb' is linear(b)
#     True
#     >>> np.zeros(3) is linear(np.zeros(3))
#     False

#     Because these are different instances.
#     """
#     return x



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
    for ix_y in numba.prange(num_y):
        for ix_x in range(num_x):
            r_squared = (ix_x-mid_x)**2 + (ix_y - mid_y)**2
            if minf2 <= r_squared <= maxf2:
                mask[ix_y, ix_x] = 1.0
            else:
                mask[ix_y, ix_x] = 0.0
    return mask



def make_closure(instance):
    """
    Creates and returns a function that simply encapsulates and
    returns the input object. Objects returned by this function can be
    used to generate original source images for the synthesis imaging
    simulation.

    Parameters
    ----------
    instance : any object
               The object to be returned from the closure function.

    Returns
    -------
    A function ``closure_fn(**kwargs)`` that, when called, simply
    returns the input object.

    Examples
    --------
    >>> a = np.zeros(3)
    >>> fn = make_closure(a)
    >>> fn()
    array([0., 0., 0.])
    >>> fn() is a
    True
    >>> fn() is np.zeros(3)
    False
    """
    def closure_fn(**_):
        return instance
    return closure_fn



def imread(filename):
    """
    Read an image from ``filename`` and return a function that returns
    that image when called. The image is read into pyfftw-created
    memory aligned arrays to enable fast FFTs using the pyfftw library.

    Parameters
    ----------
    filename : string
               The image file to read.

    Returns
    -------
    A Python function with signature fn(**kwargs) that returns  a 2D,
    byte-aligned greyscale image.

    Examples
    --------
    >>> img = imread('shirt-bw.png')
    >>> print_options = np.get_printoptions()
    >>> np.set_printoptions(precision=4)
    >>> img().shape
    (1024, 1024)
    >>> img()
    array([[0.6052, 0.6405, 0.5869, ..., 0.5007, 0.5869, 0.5346],
           [0.5882, 0.5621, 0.5987, ..., 0.5647, 0.5046, 0.5556],
           [0.5569, 0.5556, 0.434 , ..., 0.5242, 0.5281, 0.5346],
           ...,
           [0.3974, 0.4654, 0.5033, ..., 0.5163, 0.5333, 0.5242],
           [0.3961, 0.4248, 0.4928, ..., 0.5412, 0.5373, 0.5621],
           [0.3778, 0.4144, 0.5333, ..., 0.5333, 0.5046, 0.5333]],
          dtype=float32)
    >>> np.set_printoptions(**print_options)

    """
    img = np.array(plt.imread(filename)[:, :, :3].mean(axis=2), dtype='float32')
    output = pyfftw.empty_aligned(img.shape, dtype='float32')
    output[:] = img
    return make_closure(output)



@dataclass
class SimulationOutput:
    """
    Simple collection of images and associated uv-planes  for a
    synthesis imaging simulation.

    Data members
    ------------
    image : numpy.ndarray
            The original image.

    uv_plane : numpy.ndarray
               The IFFT of ``image``.

    weight : numpy.ndarray
             weight with which the original uv-plane is
             multiplied. May be complex valued.

    weighted_uv_plane : numpy.ndarray
             ``uvplane``*``weight``.

    dirty_image : numpy.ndarray
                  FFT of weighted_uv_plane.
    """
    image: np.ndarray
    uv_plane: np.ndarray
    weight: np.ndarray
    weighted_uv_plane: np.ndarray
    dirty_image: np.ndarray


def simulate(image, weights):
    '''
    image : 2D numpy.array of floats

    weights : 2D numpy.array of floats
    '''
    uv_plane = fftwpack.fftshift(fftwpack.ifft2(image, threads=4))
    weighted_uv_plane = pyfftw.empty_aligned(weights.shape, dtype='complex64')
    weighted_uv_plane[:] = weights*uv_plane
    dirty_image = fftwpack.fft2(fftwpack.fftshift(weighted_uv_plane), threads=4).real
    return SimulationOutput(image, uv_plane,
                            weights, weighted_uv_plane,
                            dirty_image)


try:
    import cv2
    def hsv_to_rgb(hsv):
        hsv = hsv.copy()
        hsv[..., 0] *= 360
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[..., ::-1]
except ImportError:
    def hsv_to_rgb(hsv):
        return mcol.hsv_to_rgb(hsv)


def complex_rgb(complex_ndarray, scale_fn=lambda x: x, perc=(3, 97), gamma=0.7):
    amps = scale_fn(np.absolute(complex_ndarray))
    min_amp, max_amp = np.percentile(amps[::20, ::20], perc)
    amps -= min_amp
    amps /= (max_amp - min_amp)
    amps[amps < 0.0] = 0.0
    amps[amps > 1.0] = 1.0
    phases = np.angle(complex_ndarray*np.exp(2j*np.pi*0.5))/(2*np.pi)+0.5
    hsv = np.empty((phases.shape[0], phases.shape[1], 3),
                   dtype=np.float32)
    hsv[..., 0] = phases
    hsv[..., 1] = 1
    hsv[..., 2] = (amps**gamma)
    return hsv_to_rgb(hsv) # mcol.hsv_to_rgb(hsv)



class FourierExplorerGui:
    def __init__(self, **kwargs):
        self.fig = plt.figure(**kwargs)
        self.axes = self.fig.subplots(2, 2)
        self.fig.subplots_adjust(hspace=0, wspace=0)
        self.imshow_artists = ((None, None), (None, None))
#        self.__name__ = 'FourierExplorerGui'


    def replot(self, scale, img_fn, min_bl_fraction, max_bl_fraction):
        ((ax_tl, ax_tr), (ax_bl, ax_br)) = self.axes
        ((ax_tl_img, ax_tr_img), (ax_bl_img, ax_br_img)) = self.imshow_artists

        mask = min_max_bl_mask(*(img_fn().shape),
                               min_bl_fraction/100.0,
                               max_bl_fraction/100.0)
        sim_result = simulate(img_fn(), mask)
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
        to_plot = scale((sim_result.uv_plane))
        #vmin, vmax = np.percentile(to_plot[::20, ::20], (1.0, 99.0))
        if ax_tr_img is None:
            ax_tr_img = ax_tr.imshow(to_plot, #vmax=vmax, vmin=vmin,
                                     interpolation='nearest')
            ax_tr.set_xticks([])
            ax_tr.set_yticks([])
        else:
            ax_tr_img.set_data(to_plot)
            #ax_tr_img.set_clim(vmin, vmax)

        # Masked uv plane
        to_plot = scale((sim_result.weighted_uv_plane))
        if ax_br_img is None:
            ax_br_img = ax_br.imshow(to_plot, #vmax=vmax, vmin=vmin,
                                     interpolation='nearest')
            ax_br.set_xticks([])
            ax_br.set_yticks([])
        else:
            ax_br_img.set_data(to_plot)
            #ax_br_img.set_clim(vmin, vmax)

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
        self.fig.canvas.draw_idle() # Apparently a bit faster than fig.canvas.draw()
        self.imshow_artists = ((ax_tl_img, ax_tr_img), (ax_bl_img, ax_br_img))


    def interact(self, image_functions):
        ipywidgets.interact(
            self.replot,
            scale=ipywidgets.fixed(complex_rgb),
            #{'Linear': linear, 'dB': dB, 'Complex': complex_rgb},
            img_fn=ipywidgets.Dropdown(options=image_functions,
                                       description='Image'),
            min_bl_fraction=ipywidgets.FloatSlider(
                min=0, max=100, step=0.25, value=0, continuous_update=True),
            max_bl_fraction=ipywidgets.FloatSlider(
                min=0, max=100, step=0.25, value=100, continuous_update=True))



pyfftw.interfaces.cache.enable()
pyfftw.interfaces.cache.set_keepalive_time(keepalive_time=100.0)
