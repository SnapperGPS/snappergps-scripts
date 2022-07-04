#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Calculate the length of a polyline through all valid GNSS fixes.

Read all JSON file in a directory assuming that they are in SnapperGPS format.
Exclude all datapoints that have no confidence.
Calculate length of polyline through all datapoints for all files.
Estimate average velocity.

Outliers needs to be removed before you run this script.

@author: Jonas Beuchert
"""
import pymap3d as pm
import numpy as np
import json
import glob
import os


def _get_dist(e, n):
    return np.sum(np.linalg.norm(np.array([np.diff(e), np.diff(n)]), axis=0))


# Enter directory with JSON files here
files = glob.glob(os.path.join("", "*.json"))

for file in files:

    print()
    print("################################################################")
    print(f"{file}")
    print("################################################################")
    print()

    # Read data file
    with open(file) as f:
        snappergps_data = json.load(f)

    # Arrays to store geodetic coordinates [decimal degrees]
    lat = [d["latitude"] for d in snappergps_data if d["confidence"] is not None]
    lon = [d["longitude"] for d in snappergps_data if d["confidence"] is not None]

    # Determine center of map
    lat0 = np.mean(lat)
    lon0 = np.mean(lon)

    # Transform geodetic coordinates into east-north-up coordinates [m]
    e, n, u = pm.geodetic2enu(np.array(lat), np.array(lon), np.zeros(len(lat)),
                              lat0, lon0, 0)

    # Get timestamps
    time = [np.datetime64(d["datetime"]) for d in snappergps_data
            if d["confidence"] is not None]
    start_datetime = time[0]

    # Make timestamps relative to start time
    time = np.array([(t-time[0]).item().total_seconds() for t in time])

    # Estimate distance using polyline
    dist = _get_dist(e, n)

    print(f"Total distance (polyline): {dist:.0f} m")
    print()

    # Estimate velocity
    vel_poly = dist / (time[-1] - time[0])
    
    # Print velocity
    print(f"Average velocity (polyline): {vel_poly:.2f} m/s")
    print()
    print(f"Average velocity (polyline): {vel_poly/1000.0*60.0*60.0:.2f} km/h")
    print()
