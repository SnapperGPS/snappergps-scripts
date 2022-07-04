# -*- coding: utf-8 -*-
"""
Trim raw SnapperGPS data to a time interval.

@author: Jonas Beuchert
"""

import json
import glob
import os
import numpy as np

# Path to JSON files (adjust)
files = glob.glob(os.path.join("", "*.json"))

# Set start & end time
start = np.datetime64('2021-08-22T10:08:00.000')
end = np.datetime64('2021-08-22T10:22:00.000')

# Loop over all files
for file in files:

    # Read JSON file
    with open(file, "r") as fp:

        data = json.load(fp)

    # Count snapshots
    n = len(data["snapshots"])

    # Get all timestamps
    dts = [np.datetime64(data["snapshots"][idx]["timestamp"])
           for idx in range(n)]

    # Get all timestamps in given interval
    good = [dt >= start and dt <= end for dt in dts]

    good_idx = np.where(good)[0]

    # Select corresponding snapshots
    new_snapshots = [data["snapshots"][idx]
                     for idx in good_idx]

    data["snapshots"] = new_snapshots

    # Save to new JSON file (pre-pend underscore)
    with open("_" + file, "w") as fp:

        json.dump(data, fp, indent=4)
