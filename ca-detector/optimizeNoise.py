from snr import signaltonoise as sn
from denoise import denoise
from waveletDenoise import waveletDenoise as wdenoise
# from waveletDenoise import waveletDenoiseTest
from statistics import mean
import os
import numpy as np
import pywt

os.chdir('ecgiddb')
orig = []
snrs = []
snrs_wv = []
sns = []
for norm in os.listdir():
    tfile = open(norm, "r")
    lines = (tfile.read().splitlines())
    arr = np.float64(lines)
    xnorm = denoise(arr)
    wnorm = wdenoise(arr)

    orig.append(sn(arr))
    snrs.append(sn(xnorm))
    snrs_wv.append(sn(wnorm))

#     trials = waveletDenoiseTest(arr)
#     s = mean(orig)
#     for t in trials:
#         m = np.mean(sn(t))
#         snval = abs(m*100/(s-m))
#         sns.append(snval)
#
# highest = list(pywt.wavelist(kind='discrete'))[np.argmax(sns)]
# print(np.argmax(sns), highest, sns[np.argmax(sns)])

s = mean(orig)
m1 = mean(snrs)
m2 = mean(snrs_wv)
sn1 = abs(m1*100/(s-m1))
sn2 = abs(m2*100/(s-m2))
print("Mean signaltonoise for noisy array: " + str(s))
print("Mean signaltonoise for ecgiddb and original: " + str(m1) + " and SNR of: " + str(sn1))
print("Mean signaltonoise for ecgiddb and wavelets: " + str(m2) + " and SNR of: " + str(sn2))
