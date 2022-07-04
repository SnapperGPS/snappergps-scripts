#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Remove fixes with no confidences and temperature and battery from CSV file.

@author: Jonas Beuchert
"""
import glob
import numpy as np

# Path to files goes here
files = glob.glob("*.csv")

for file in files:
    datetimes, lats, lons, confidences = np.loadtxt(file,
                                                    delimiter=',',
                                                    skiprows=1,
                                                    usecols=[0, 1, 2, 3],
                                                    unpack=True,
                                                    dtype="str")
    datetimes = datetimes[confidences != ""]
    lats = lats[confidences != ""]
    lons = lons[confidences != ""]
    confidences = confidences[confidences != ""]
    np.savetxt(f"{file[:-4]}_reduced.csv",
               np.array([datetimes, lats, lons, confidences]).T,
               newline="\n",
               delimiter=",",
               header="datetime,latitude,longitude,confidence",
               fmt="%s",
               comments="")
            