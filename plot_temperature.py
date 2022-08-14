#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Read a SnapperGPS JSON file and plot the temperature over time.

@author: Jonas Beuchert
"""
import json
import matplotlib.pyplot as plt
import numpy as np


# Set the path to your JSON file here (downloaded from https://snappergps.info/view)
snappergps_file = "aabbccddee.json"

with open(snappergps_file) as f:
    snappergps_data = json.load(f)

# Create temperature list
temp = [d["temperature"] for d in snappergps_data]

# Create timestamp list
time = [np.datetime64(d["datetime"]) for d in snappergps_data]

# Plot temepratures over timestamps
plt.figure(figsize=[12.0, 4.8])
plt.plot(time, temp)
plt.xlabel("time")
plt.ylabel("T [°C]")
plt.grid()
plt.show()
