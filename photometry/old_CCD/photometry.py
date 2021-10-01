#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Scripts for the "Técnicas Observacionales en Astrofísica" (TOA) M.Sc. course
Also available to "TFG-EXP Astro" (B.Sc.)

@author: Yago Ascasibar (UAM)
Created on Thu Oct  5 09:58:10 2017
"""

from astropy.io import fits
import numpy as np
from matplotlib import pyplot as plt


# %% Function definitions

def read_file(filename):
    """Open FITS file and return (data, exposure time) as a tuple."""
    hdu = fits.open(filename)  # HDU = Header/Data Unit (standard FITS nomenclature)

#    Print basic information about the file:
    # hdu.info()
    # for extension in hdu:
    #     print('--- ', extension.name, ' ---')
    #     hdr = extension.header
    #     for key in hdr:
    #         print(key, '\t', hdr[key])

    return hdu[0].data, hdu[0].header['EXPTIME']  # counts/pixel, exposure time


def show_data(data, title=''):
    """Display image with adaptive color bar."""
    plt.figure(figsize=(15, 6))
    plt.imshow(data, origin='lower',
               vmin=np.percentile(data, 5), vmax=np.percentile(data, 95))
    plt.colorbar()
    plt.title(title)
    plt.show()
    plt.close()


# %% Bias

bias = read_file('Bias.fits')[0]
show_data(bias, 'Bias (counts)')

# %% Dark

dark_counts, t_dark = read_file('Dark.fits')
dark_current = (dark_counts-bias) / t_dark
show_data(dark_current, 'Dark (counts/s)')

# %% Flat

flat_field, t_flat = read_file('Flat_V.fits')
flat_field -= bias  # subtract bias
flat_field -= dark_current*t_flat  # subtract dark
flat_field /= np.nanmedian(flat_field)
show_data(flat_field, 'Flat field (normalised)')

# %% Raw image

object_counts, t_object = read_file('M31.fits')
show_data(object_counts, 'Raw counts')

# %% Reduced image

show_data(object_counts - bias - dark_current*t_object,
          'Raw counts - bias - dark (counts)')

object_rate = (object_counts - bias)/t_object - dark_current
object_rate /= flat_field

show_data(object_rate, 'Reduced rate (counts/s)')


# %% Sky subtraction

plt.title('Histogram of reduced counts')
scaled_value = object_rate.flatten()
v0 = np.nanpercentile(scaled_value, 10)
v1 = np.nanpercentile(scaled_value, 90)
L = v1-v0
bins = np.linspace(v0-L, v1+L, 100)
histogram = plt.hist(scaled_value, bins=bins)

# Find peak
sky_intensity = bins[np.argmax(histogram[0])]
plt.axvline(sky_intensity, 0, np.max(histogram[0]), c='b')
plt.text(sky_intensity, np.max(histogram[0]),
         ' sky intensity: {:.2f} counts'.format(sky_intensity))
plt.show()

# Subtract
sky_subtracted = object_rate - sky_intensity

plt.title('Histogram of sky-subtracted counts')
scaled_value = sky_subtracted.flatten()
v0 = np.nanpercentile(scaled_value, 10)
v1 = np.nanpercentile(scaled_value, 90)
L = v1-v0
bins = np.linspace(v0-L, v1+L, 100)
histogram = plt.hist(scaled_value, bins=bins)
plt.axvline(0, 0, np.max(histogram[0]), c='b')
plt.show()

show_data(sky_subtracted, 'Sky-subtracted counts')

# -----------------------------------------------------------------------------
#                                                    ... Paranoy@ Rulz! ;^D
# -----------------------------------------------------------------------------
