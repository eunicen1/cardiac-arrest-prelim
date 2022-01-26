from psnr import signaltonoise as snr
from denoise import denoise
from waveletDenoise import waveletDenoise as wdenoise
from statistics import mean
import os
import numpy as np

os.chdir('ecgiddb')
orig = []
snrs = []
snrs_wv = []
for norm in os.listdir():
    tfile = open(norm, "r")
    lines = (tfile.read().splitlines())
    arr = np.float64(lines)
    xnorm = denoise(arr)
    wnorm = wdenoise(arr)
    orig.append(snr(arr))
    snrs.append(snr(xnorm))
    snrs_wv.append(snr(wnorm))

print(orig, snrs, snrs_wv)
s = mean(orig)
m1 = mean(snrs)
m2 = mean(snrs_wv)
sn1 = m1*100/(s-m1) # should I abs()?
sn2 = m2*100/(s-m2) # should I abs()?
print("Mean signaltonoise for noisy array: " + str(s))
print("Mean signaltonoise for ecgiddb and original: " + str(m1) + " and SNR of: " + str(sn1))
print("Mean signaltonoise for ecgiddb and wavelets: " + str(m2) + " and SNR of: " + str(sn2))
