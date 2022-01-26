import numpy as np
# https://datagy.io/mean-squared-error-python/

# def psnr(orig, deno):
#     mseval = mse(orig, deno)
#     return 10*np.log10(((len(orig)-1)**2)/mseval)
#
# def mse(orig, dat):
#     return np.sqrt(np.sum((orig-dat)**2))
def signaltonoise(a, axis=0, ddof=0):
    # https://stackoverflow.com/questions/64227316/how-to-calculate-snr-of-a-image-in-python
    a = np.asanyarray(a)
    m = a.mean(axis)
    sd = a.std(axis=axis, ddof=ddof)
    return np.float64(np.where(sd == 0, 0, m/sd))
