from ifrsim import noise
from astropy import units as u
import numpy as np


def test_rms_voltage_from_power():
    rmsvfp = noise.rms_voltage_from_power
    
    assert rmsvfp(2*u.W).unit == u.V
    try:
        rmsvfp(2*u.m**0.5)
    except u.UnitConversionError:
        pass

def test_gauss_noise():
    bw = 2.400*u.MHz
    dt = 10*u.s
    p_nu = 10000*u.Jy*0.03*u.m**2
    sample_interval = 1/(2.0*bw)
    num_samp = int(np.ceil(dt/sample_interval))
    result = noise.gauss_noise(bw, dt, p_nu)
    assert result.shape == (num_samp,)
    assert result.dtype == np.float64
    print(bw, dt, 'n = ', num_samp)
    print(result.std(), np.sqrt((result**2).mean()))
    print((result.std()**2/(50*u.ohm)).to(u.W))
