# -*- coding: utf-8 -*-
"""
Plot accelerometer data from SnapperGPS JSON files.

@author: Jonas Beuchert
"""

import numpy as np
import matplotlib.pyplot as plt
import json
import os


# Set name of directory and file here
directory = "test_data"
file = "24E9D2085FE529EF_20230717_192625_002.json"

# Assemble path
path = os.path.join(directory, file)

# Read JSON file
with open(path, "r") as fp:
    
        data = json.load(fp)

# Get fields acceleration_x, acceleration_y, acceleration_z
acceleration_x = [datapoint["acceleration_x"]
                    for datapoint in data["snapshots"]]
acceleration_y = [datapoint["acceleration_y"]
                    for datapoint in data["snapshots"]]
acceleration_z = [datapoint["acceleration_z"]
                    for datapoint in data["snapshots"]]
# Get timestamps
timestamps = [np.datetime64(datapoint["timestamp"])
                for datapoint in data["snapshots"]]

# Calulate norm of accelerations
acceleration_norm = np.linalg.norm(np.array([acceleration_x,
                                             acceleration_y,
                                             acceleration_z]), axis=0)

timestamps = np.arange(len(acceleration_x))

# Plot accelerations over time
plt.figure()
plt.plot(timestamps, acceleration_x, label="x")
plt.plot(timestamps, acceleration_y, label="y")
plt.plot(timestamps, acceleration_z, label="z")
plt.plot(timestamps, acceleration_norm, label="norm")
plt.xlabel("time")
plt.ylabel("acceleration [g]")
plt.grid()
plt.legend()
plt.show()
