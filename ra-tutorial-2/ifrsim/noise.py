r'''
This module contains functions to simulate noise in a radio interferometer.
'''

import numpy as np
from astropy import units as u
from scipy import constants as const


def normalized_gaussian_noise(num_samples: int, dtype=np.float64, seed=None):
    r'''Produce num_samples normally-distributed random numbers with zero
    mean and unit standard deviation. 
    
    **Parameters**
    
    num_samples : int
        Number of samples to draw

    dtype: numpy.dtype
        Data type of the output array

    seed : {None, int, array_like[ints], ISeedSequence, BitGenerator, Generator}, optional
        (Via numpy documentation): A seed to initialize the
        BitGenerator. If None, then fresh, unpredictable entropy will
        be pulled from the OS. If an int or array_like[ints] is
        passed, then it will be passed to SeedSequence to derive the
        initial BitGenerator state. One may also pass in an
        implementor of the ISeedSequence interface like
        SeedSequence. Additionally, when passed a BitGenerator, it
        will be wrapped by Generator. If passed a Generator, it will
        be returned unaltered.


    **Returns**

    One-dimensional ``numpy.array`` of length ``num_samples`` and type
    ``dtype``.


    **Examples**
    
    >>> num_samples = 13
    >>> dtype = np.float32
    >>> result = normalized_gaussian_noise(num_samples, dtype)
    >>> result.dtype
    dtype('float32')
    >>> result.shape
    (13,)
    >>> n = 100000000
    >>> large_set = normalized_gaussian_noise(n, dtype=np.float64)
    >>> sigma_mean = 1/np.sqrt(n)
    >>> sigma_mean
    0.0001

    Verify that mean is consistent with zero:
    >>> large_set.mean() < 5*sigma_mean
    True

    Verify that standard deviation is consistent with 1.0:
    >>> np.abs(large_set.std() - 1) < 5*sigma_mean
    True

    Verify that standard deviation == RMS:
    >>> np.abs(large_set.std()-(large_set**2).mean()) < 5*sigma_mean
    True

    '''
    rng = np.random.default_rng(seed)
    voltages_normalized = rng.standard_normal(size=num_samples, dtype=dtype)
    return voltages_normalized





def digitize(real_sequence: 'np.ndarray', nrbits:int=8):
    r'''
    Digitize a floating point sequence by rounding to the nearest
    integer. Uses at most nrbits and clips excess values at
    2**(nrbits-1) -1 and -(2**(nrbits-1)). This function simulates and
    ideal analogue to digital converter (ADC).
    
    **Parameters**

    real_sequence : numpy.ndarray
        Real numbers to sample. Must be a floating point number type.

    nrbits : int
        maximum number of bits at which the "ADC" samples.


    **Returns**

    np.ndarray[int]


    **Examples**

    >>> noise_float = 2*normalized_gaussian_noise(num_samples=40, seed=7)
    >>> noise_float
    array([ 2.46030671e-03,  5.97491075e-01, -5.48275711e-01, -1.78118368e+00,
           -9.09341570e-01, -1.98329311e+00,  1.20287205e-01,  2.68043049e+00,
           -9.84413037e-01, -1.24094980e+00,  9.79684100e-01,  7.13774016e-01,
            2.10828498e-01, -1.86093609e+00, -5.85036449e-02,  1.39060639e+00,
           -2.68842909e+00, -9.15231522e-01, -3.80244548e+00, -2.57907548e+00,
           -3.68347008e+00, -4.70182262e-01, -2.53489296e+00,  5.42528718e-01,
            3.13502173e-01, -3.73861889e-01, -5.03351942e+00, -1.07738579e+00,
           -9.70018908e-02,  2.26617972e-01, -3.06027153e+00, -9.55506552e-01,
           -1.95703816e+00, -1.61767448e+00,  2.12179725e+00, -1.61506935e+00,
           -6.50434099e-02,  1.76877973e+00, -1.16720087e+00, -2.23403899e-01])
    >>> noise_int = digitize(noise_float, nrbits=3)
    >>> noise_int
    array([ 0,  1, -1, -2, -1, -2,  0,  3, -1, -1,  1,  1,  0, -2,  0,  1, -3,
           -1, -4, -3, -4,  0, -3,  1,  0,  0, -4, -1,  0,  0, -3, -1, -2, -2,
            2, -2,  0,  2, -1,  0])

    '''
    result = np.rint(real_sequence).astype(int)
    maxpos = 2**(nrbits-1) -1
    maxneg = -(2**(nrbits-1))
    result[result < maxneg] = maxneg
    result[result > maxpos] = maxpos
    return result



def rms_voltage_from_power(p, impedance=50*u.Ohm):
    r'''
    Return the RMS voltage of random noise in a circuit detected by an
    impedance-matched voltage meter.
    
    **Parameters**
    
    **Returns**
    
    Value (scalar or when appropriate np.array) in units of astropy.units.V
    '''
    return np.sqrt(p*impedance).to(u.V)


def gauss_noise(bandwidth, duration, power_density, impedance=50*u.ohm):
    '''
    This function simulates a sequence of thermal noise with the provided band width 
    and at the requested duration. The simulated time series starts at the beginning
    of the first sample and ends within the last sample. 
    '''
    sample_interval = 1/(2.0*bandwidth)
    num_samples = int(np.ceil(duration/sample_interval))
    voltages_normalized = normalized_gaussian_noise(num_samples)
    sigma = np.sqrt(bandwidth*power_density*impedance)
    return (voltages_normalized*sigma).to(u.uV)


def matched_resistor_power_density(temperature):
    r'''
    '''
    k_b = const.k*u.J/u.K
    return (k_b*temperature).to(u.W/u.Hz)


