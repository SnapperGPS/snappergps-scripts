#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Use Gaussian process regression to smooth GNSS data and various calculations.

Read all JSON file in a directory assuming that they are in SnapperGPS format.
Exclude all datapoints that have no confidence.
Fit Gaussian process to data.
Calculate length of smoothed track.
Calculate length of track for each day.
Estimate average velocity.
Plot smoothed track.
Save smoothed track to animated KML file.
Save smoothed track in a KML file with an alternative style.
Estimate and plot area covered by smoothed track.

Outliers needs to be removed before you run this script.

@author: Jonas Beuchert
"""
import pymap3d as pm
import numpy as np
import matplotlib.pyplot as plt
import json
import glob
import os

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, Matern

from scipy.spatial import ConvexHull


# Set this to create a KML file
create_kml = True

def _get_dist(e, n):
    return np.sum(np.linalg.norm(np.array([np.diff(e), np.diff(n)]), axis=0))


def _plot_gp(time, y, confidence, y_pred, sigma, x):
    plt.errorbar(time, y, confidence / np.sqrt(2), fmt="r.",
                 label="Observations")
    plt.plot(x, y_pred, "b-", label="Prediction")
    plt.fill(
        np.concatenate([x, x[::-1]]),
        np.concatenate([y_pred - 1.9600 * sigma,
                        (y_pred + 1.9600 * sigma)[::-1]]),
        alpha=0.5,
        fc="b",
        ec="None",
        label="95% confidence interval",
    )
    plt.grid()
    plt.xlabel("time [s]")


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

    # Get uncertainty
    confidence = np.array([d["confidence"] for d in snappergps_data
                           if d["confidence"] is not None])

    # Mesh the input space for the prediction
    x = np.atleast_2d(np.arange(start=time[0], stop=np.ceil(time[-1]), step=60)).T

    # Use time as input variable for Gaussian Process
    X = np.atleast_2d(time).T

    # Kernel for Gaussian Process model (try different ones)
    # kernel = ConstantKernel(1.0, (1e-3, 1e3)) *  RBF(10, (1, 1e3))
    # kernel = ConstantKernel(1e4, (1e3, 1e10)) * Matern(60*60, (60*60, 24*60*60))
    # kernel = ConstantKernel() * Matern()
    # kernel = ConstantKernel(1e6, (1e5, 1e10)) * Matern(60*60, (60*60, 24*60*60))
    kernel = ConstantKernel(1.14e+04**2, constant_value_bounds="fixed") * Matern(5.79e+04, length_scale_bounds="fixed")

    # Instantiate a Gaussian Process model
    gp = GaussianProcessRegressor(kernel=kernel,
                                  alpha=(confidence/np.sqrt(2)) ** 2,
                                  n_restarts_optimizer=10)

    # Fit to data using Maximum Likelihood Estimation of the parameters
    print("Fit...")
    gp.fit(X, np.array([e, n]).T)
    print(gp.kernel_)

    print("Predict...")
    # Make the prediction on the meshed x-axis (ask for MSE as well)
    y_pred, sigma = gp.predict(x, return_std=True)

    e_pred = y_pred[:, 0]
    n_pred = y_pred[:, 1]
    e_sigma = sigma
    n_sigma = sigma

    # Calculate arc length
    dist = _get_dist(e_pred, n_pred)

    print()
    print(f"Total distance (GP): {dist:.0f} m")
    print()

    # Estimate velocity
    vel_gp = dist / (time[-1] - time[0])

    # Restore dates of timestamps
    x_full = (start_datetime + np.array([np.timedelta64(int(x_idx), "s")
                                         for x_idx in x[:, 0]])).astype("M8[D]")

    # Get unique dates
    x_unique = np.unique(x_full)

    # Get travel distance for each day
    for day in x_unique:
        idx_day = np.where(x_full == day)[0]
        dist_day = _get_dist(e_pred[idx_day], n_pred[idx_day])
        print(f"Distance on {day} (GP): {dist_day:.0f} m")
    print()

    # Plot track
    fig, ax = plt.subplots()
    plt.plot(e, n, "*-")
    plt.plot(e_pred, n_pred)
    circle_0 = plt.Circle((e_pred[0], n_pred[0]), sigma[0] * 1.96 * np.sqrt(2),
                          fc=(0.5, 0.5, 1.0), alpha=1.0, ec=None)
    plt.gca().add_patch(circle_0)
    circle_1 = plt.Circle((e_pred[-1], n_pred[-1]), sigma[-1] * 1.96 * np.sqrt(2),
                          fc=(0.5, 0.5, 1.0), alpha=1.0, ec=None)
    plt.gca().add_patch(circle_1)
    plt.grid()
    plt.xlabel("east [m]")
    plt.ylabel("north [m]")
    x_lim = ax.get_xlim()
    y_lim = ax.get_ylim()
    ax.set_aspect('equal', adjustable='box')
    plt.title(file.split("/")[-1].split(".")[0])
    plt.show()

    # Plot GP
    fig, ax = plt.subplots()
    plt.subplot(2, 1, 1)
    _plot_gp(time, e, confidence, e_pred, e_sigma, x)
    plt.ylabel("east [m]")
    plt.title(file.split("/")[-1].split(".")[0])
    plt.subplot(2, 1, 2)
    _plot_gp(time, n, confidence, n_pred, n_sigma, x)
    plt.ylabel("north [m]")
    plt.title(file.split("/")[-1].split(".")[0])
    plt.tight_layout()
    plt.show()

    if create_kml:
        print("Write KML file...")
        step = 10
        lat_pred, lon_pred, _ = pm.enu2geodetic(e_pred, n_pred,
                                                np.zeros(len(e_pred)),
                                                lat0, lon0, 0.0)
        with open(f"{file[:-5]}_animated.kml", "w") as file_pointer:
            file_pointer.write(f"""<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2" xmlns:gx="http://www.google.com/kml/ext/2.2" xmlns:kml="http://www.opengis.net/kml/2.2" xmlns:atom="http://www.w3.org/2005/Atom">
    <Document>
        <name>Animated SnapperGPS track {file[:-5]}</name>
        <description></description>
        <open>1</open>

        <Style id="line-style">
            <LineStyle>
                <color>bf00aaff</color>      <!-- this is the color of your path -->
                <width>5</width>        <!-- this is the width of your path -->
            </LineStyle>
        </Style>

        <!-- this is the camera view  -->

        <LookAt>
            <longitude>{lon0}</longitude>
            <latitude>{lat0}</latitude>
            <altitude>0</altitude>
            <heading>0</heading>
            <tilt>40</tilt>
            <range>73000</range>
            <gx:altitudeMode>relativeToSeaFloor</gx:altitudeMode>
        </LookAt>

        <gx:Tour>
            <name>Double-click here to start tour</name>
            <gx:Playlist>

                 <gx:Wait> <gx:duration>1</gx:duration></gx:Wait> <!-- short pause at the beginning -->

                 <!-- line animation -->

""")
            for idx in range(0, len(lat_pred) - step, step):
                file_pointer.write(f"""                 <gx:AnimatedUpdate>
                    <Update>
                        <Change><Placemark targetId="{int(idx / step)}"><visibility>1</visibility></Placemark></Change>
                    </Update>
                </gx:AnimatedUpdate>

                <gx:Wait><gx:duration>0.02</gx:duration></gx:Wait>   <!-- this is the length of time between path segments coming on, longer time will be a slower animation -->

""")
            file_pointer.write("""            </gx:Playlist>
        </gx:Tour>

        <!-- the tour ends here and the following is the line information -->

        <Folder>
            <name>Path segments</name>

            <Style>
                <ListStyle>
                    <listItemType>checkHideChildren</listItemType>
                </ListStyle>
            </Style>

""")
            for idx in range(0, len(lat_pred) - step, step):
                file_pointer.write(f"""            <Placemark id="{int(idx / step) + 1}">
                <name>{int(idx / step) + 1}</name>
                <visibility>0</visibility>
                <styleUrl>#line-style</styleUrl>
                <LineString>
                    <tessellate>1</tessellate>
                    <coordinates>
                        {lon_pred[idx]},{lat_pred[idx]},0 {lon_pred[idx + step]},{lat_pred[idx + step]},0
                    </coordinates>
                </LineString>
            </Placemark>
""")
            file_pointer.write("""
        </Folder>

    </Document>
</kml>
""")
        print("KML file saved.")
        print()
        print("Write KML file in another style...")

        with open(f"{file[:-5]}_smoothed.kml", "w") as file_pointer:
            file_pointer.write(f"""<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
    <Document id="root_doc">
        <Schema name="{file[:-5]}_KML" id="{file[:-5]}_KML">
            <SimpleField name="descriptio" type="string"></SimpleField>
            <SimpleField name="timestamp" type="string"></SimpleField>
            <SimpleField name="begin" type="string"></SimpleField>
            <SimpleField name="end" type="string"></SimpleField>
            <SimpleField name="altitudeMo" type="string"></SimpleField>
            <SimpleField name="tessellate" type="float"></SimpleField>
            <SimpleField name="extrude" type="float"></SimpleField>
            <SimpleField name="visibility" type="float"></SimpleField>
            <SimpleField name="drawOrder" type="float"></SimpleField>
            <SimpleField name="icon" type="string"></SimpleField>
            <SimpleField name="snippet" type="string"></SimpleField>
        </Schema>
        <Folder>
            <name>{file[:-5]}_KML</name>
            <Placemark>
                <name>SnapperGPS data</name>
                <Style>
                    <LineStyle>
                        <color>ff0000ff</color>
                    </LineStyle>
                    <PolyStyle>
                        <fill>0</fill>
                    </PolyStyle>
                </Style>
                <ExtendedData>
                    <SchemaData schemaUrl="#{file[:-5]}_KML">
                        <SimpleData name="descriptio">&amp;nbsp;</SimpleData>
                        <SimpleData name="altitudeMo">clampToGround</SimpleData>
                        <SimpleData name="tessellate">1</SimpleData>
                        <SimpleData name="extrude">0</SimpleData>
                        <SimpleData name="visibility">-1</SimpleData>
                    </SchemaData>
                </ExtendedData>
                <MultiGeometry>
                    <LineString>
                        <coordinates>
""")
            for idx in range(0, len(lat_pred) - step, step):
                file_pointer.write(f"""                            {lon_pred[idx]},{lat_pred[idx]}
""")
            file_pointer.write("""                        </coordinates>
                    </LineString>
                </MultiGeometry>
            </Placemark>
        </Folder>
    </Document>
</kml>
""")

        print("KML file saved.")
        print()

    # Get area of convex hull of track

    # Array of 2D points
    points = np.array([e_pred, n_pred]).transpose()

    # Convex hull
    hull = ConvexHull(points)

    # Plot points and convex hull
    fig, ax = plt.subplots()
    plt.plot(points[:, 0], points[:, 1], "o")
    plt.plot(points[hull.vertices, 0], points[hull.vertices, 1], "r--", lw=2)
    plt.plot(points[hull.vertices[0], 0], points[hull.vertices[0], 1], "ro")
    plt.xlabel("east [m]")
    plt.ylabel("north [m]")
    x_lim = ax.get_xlim()
    y_lim = ax.get_ylim()
    ax.set_aspect('equal', adjustable='box')
    plt.title(file.split("/")[-1].split(".")[0])
    plt.show()

    # Print area
    print(f"Area covered by the track (GP): {hull.volume:.0f} m^2")
    print()

    # Print velocity
    print(f"Average velocity (GP): {vel_gp:.2f} m/s")
    print()
    print(f"Average velocity (GP): {vel_gp/1000.0*60.0*60.0:.2f} km/h")
    print()
