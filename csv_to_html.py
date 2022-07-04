
# -*- coding: utf-8 -*-
"""
Plot all SnapperGPS tracks in a folder on an HTML map.

Expect SnapperGPS tracks in CSV files.
Exclude fixes without confidence.
Map can be displayed in a browser.

@author: Jonas Beuchert
"""

import folium
import numpy as np
import glob
import os

# Centre of map (latitude and longitude in decimal degrees)
pos_ref_geo = np.array([15.215534, -23.152380])

# Create map
m = folium.Map(
        location=[pos_ref_geo[0], pos_ref_geo[1]],
        # tiles='http://tile.stamen.com/terrain-background/{z}/{x}/{y}.png',
        attr=' ',
        control_scale=True,
        zoom_control=False,
        tiles='OpenStreetMap',  # 'Stamen Toner' 'Stamen Terrain' 'Mapbox Bright' 'Mapbox Control Room'
        zoom_start=13
    )

# Files to display (adjust path to files, if necessary)
files = glob.glob(os.path.join("", "*.csv"))

# Color map
colors = [
    'red',
    'blue',
    'black',
    'darkred',
    'lightred',
    'purple',
    'orange',
    'beige',
    'green',
    'darkgreen',
    'darkblue',
    'lightgreen',
    'darkpurple',
    'pink',
    'cadetblue',
    'lightgray',
    'black'
]
colors = colors[:len(files)]

# Loop over all CSV files
for path, color in zip(files, colors):

    label = path[:-4]

    print(f"Process file {label}.")

    if len(path) > 0:

        # Load estimated positions
        lats, lons = np.loadtxt(path, delimiter=',',
                                skiprows=1, usecols=[1, 2], unpack=True)

        confidences = np.loadtxt(path, delimiter=',', dtype=str,
                                 skiprows=1, usecols=[3], unpack=True)

        lats = lats[confidences != ""]
        lons = lons[confidences != ""]

    track = [(lat, lon) for lat, lon in zip(lats, lons)]

    # Draw track as polyline
    folium.PolyLine(track,
                    color=color,
                    weight=3,
                    opacity=0.4,
                    popup=label,
                    # dash_array='10'
                    ).add_to(m)

# Save map as HTML file
m.save("all_tracks.html")
print('Saved locations on map as HTML file.')
print()
