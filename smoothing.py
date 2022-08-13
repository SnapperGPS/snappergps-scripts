# -*- coding: utf-8 -*-
"""
Apply smoothing to a SnapperGPS file and compare with ground truth.

@author: Jonas Beuchert
"""
from pyubx2 import UBXReader
import pymap3d as pm
import numpy as np
import matplotlib.pyplot as plt
import json
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, Matern


###############################################################################
# Use the following lines for a short test
###############################################################################

snappergps_file = "test_data/a37e32c16e.json"
# Snapshot interval
dT = 1.0
# Start time (minutes)
start_min = 40
# Manhatten distance of fix from origin to be considered an outlier [m]
outlier_threshold = np.inf
# Ground truth data from .ubx file
gt_file = "test_data/COM6_211122_123733.ubx"

###############################################################################
# Use the following lines for a long test
###############################################################################

# snappergps_file = "test_data/8418a77733.json"
# # Snapshot interval
# dT = 1.0
# # Start time (minutes)
# start_min = 30
# # Manhatten distance of fix from origin to be considered an outlier [m]
# outlier_threshold = 750.0
# # Ground truth data from .ubx file
# gt_file = "test_data/COM6_211125_082553.ubx"

###############################################################################
# Data loading
###############################################################################

# Read data file
with open(snappergps_file) as f:
    snappergps_data = json.load(f)

# Arrays to store geodetic coordinates [decimal degrees]
lat = [d["latitude"] for d in snappergps_data]
lon = [d["longitude"] for d in snappergps_data]

# Use center of track as origin of map / ENU coordinates
lat0 = np.median(lat)
lon0 = np.median(lon)

# Transform geodetic coordinates into east-north-up coordinates [m]
e, n, u = pm.geodetic2enu(np.array(lat), np.array(lon), np.zeros(len(lat)),
                          lat0, lon0, 0)

# Get timestamps
time = [np.datetime64(d["datetime"]) for d in snappergps_data]
# Get time differences
dt = np.concatenate((np.array([np.nan]),
                     np.array([t.item().total_seconds()
                               for t in np.diff(time)])
                     ))

# Get uncertainties
confidence = [d["confidence"] if d["confidence"] is not None else np.inf
              for d in snappergps_data]

###############################################################################
# Pre-processing
###############################################################################

# Crude outlier rejection
confidence = np.array(confidence)
confidence[np.where(np.logical_or(np.abs(e) >= outlier_threshold,
                                  np.abs(n) >= outlier_threshold))[0]] = np.inf

###############################################################################
# Extended Kalman filter and extended Rauch-Tung-Striebel smoother
###############################################################################

# Observations
z = np.array([dt, e, n]).transpose()


def _f(x):
    """State transition function."""
    return np.array([x[0],
                     x[1] + x[0] * x[3],
                     x[2] + x[0] * x[4],
                     x[3],
                     x[4]])


def _F(x):
    """Jacobian of state transition function."""
    return np.array([[1, 0, 0, 0, 0],
                     [x[3], 1, 0, x[0], 0],
                     [x[4], 0, 1, 0, x[0]],
                     [0, 0, 0, 1, 0],
                     [0, 0, 0, 0, 1]])


# Observation model
H = np.array([[1, 0, 0, 0, 0],
              [0, 1, 0, 0, 0],
              [0, 0, 1, 0, 0]])
# Covariance of process noise
Q = np.diag([0.0003, 0.02, 0.02, 0.0028, 0.0028])
# Covariance of observation noise
R = np.array([np.diag([1e-9, c / np.sqrt(2), c / np.sqrt(2)])
              for c in confidence])

# A priori state estimate at time k given observations up to and including at
# time k-1
x_prio = np.full((len(time), 5), np.nan)
# A priori estimate covariance matrix (a measure of the estimated accuracy of
# the state estimate)
P_prio = np.full((len(time), 5, 5), np.nan)
# A posteriori state estimate at time k given observations up to and including
# at time k
x_post = np.full((len(time), 5), np.nan)
# A posteriori estimate covariance matrix (a measure of the estimated accuracy
# of the state estimate)
P_post = np.full((len(time), 5, 5), np.nan)

# Initialise with 1st observation
x_post[0] = np.array([dT, e[0], n[0], 0.0, 0.0])
P_post[0] = np.diag([0.1,
                     confidence[0] / np.sqrt(2), confidence[0] / np.sqrt(2),
                     5.0, 5.0])

# Forward pass, same as regular extended Kalman filter
for k in range(1, len(time)):
    # Predicted (a priori) state estimate
    x_prio[k] = _f(x_post[k-1])  # EKF
    # Predicted (a priori) estimate covariance
    F = _F(x_post[k-1])  # EKF
    P_prio[k] = F @ (P_post[k-1]) @ F.T + Q
    # Check if observation is valid
    if np.inf not in R[k]:
        # Innovation or measurement pre-fit residual
        y = z[k] - H @ x_prio[k]
        # Innovation (or pre-fit residual) covariance
        S = H @ P_prio[k] @ H.T + R[k]
        # Optimal Kalman gain
        K = P_prio[k] @ H.T @ np.linalg.inv(S)
        # Updated (a posteriori) state estimate
        x_post[k] = x_prio[k] + K @ y
        # Updated (a posteriori) estimate covariance
        P_post[k] = (np.eye(5) - K @ H) @ P_prio[k]
        # Measurement post-fit residual
        # y = z[k] - H @ x_pos[k]
    else:
        # Ignore invalid observation, just propagate state
        x_post[k] = x_prio[k]
        P_post[k] = P_prio[k]

# Smoothed state estimates
x_smooth = np.full((len(time), 5), np.nan)
# Smoothed covariances
P_smooth = np.full((len(time), 5, 5), np.nan)

# Initialise with last filtered estimate
x_smooth[-1] = x_post[-1]
P_smooth[-1] = P_post[-1]

# Backward pass
for k in range(len(time)-2, -1, -1):
    F = _F(x_post[k+1])  # EKF
    C = P_post[k] @ F.T @ np.linalg.inv(P_prio[k+1])
    x_smooth[k] = x_post[k] + C @ (x_smooth[k+1] - x_prio[k+1])
    P_smooth[k] = P_post[k] + C @ (P_smooth[k+1] - P_prio[k+1]) @ C.T

###############################################################################
# Gaussian process regression
###############################################################################


def _gpr(snappergps_data, timestamps):
    """Gaussian process regression."""
    # Arrays to store geodetic coordinates [decimal degrees]
    lat = [d["latitude"] for d in snappergps_data
           if d["confidence"] is not None]
    lon = [d["longitude"] for d in snappergps_data
           if d["confidence"] is not None]

    # Transform geodetic coordinates into east-north-up coordinates [m]
    e, n, u = pm.geodetic2enu(np.array(lat), np.array(lon), np.zeros(len(lat)),
                              lat0, lon0, 0)

    # Get timestamps
    time = [np.datetime64(d["datetime"]) for d in snappergps_data
            if d["confidence"] is not None]

    # Make timestamps relative to start time
    time = np.array([(t-time[0]).item().total_seconds() for t in time])

    # Get uncertainty
    confidence = np.array([d["confidence"] for d in snappergps_data
                           if d["confidence"] is not None])

    # Crude outlier rejection
    good_idx = np.where(np.logical_and(np.abs(e) < outlier_threshold,
                                       np.abs(n) < outlier_threshold))[0]
    e = e[good_idx]
    n = n[good_idx]
    time = time[good_idx]
    confidence = np.array(confidence)[good_idx]

    # Mesh the input space for the prediction
    x = np.atleast_2d(timestamps).T

    # Use time as input variable for Gaussian Process
    X = np.atleast_2d(time).T

    # Kernel for Gaussian Process model
    # kernel = ConstantKernel(1.0, (1e-3, 1e3)) * RBF(10, (1, 1e3))
    kernel = ConstantKernel() * Matern()

    # Instantiate a Gaussian Process model
    gp = GaussianProcessRegressor(kernel=kernel,
                                  alpha=(confidence/np.sqrt(2)) ** 2,
                                  n_restarts_optimizer=10)

    # Fit to data using Maximum Likelihood Estimation of the parameters
    print("Fit...")
    gp.fit(X, np.array([e, n]).T)
    print(gp.kernel_)

    print("Predict...")
    # Make the prediction on the meshed x-axis
    return gp.predict(x, return_std=False)


x_gpr = _gpr(snappergps_data,
             [(np.datetime64(d["datetime"])
               - np.datetime64(snappergps_data[0]["datetime"])
               ).item().total_seconds() for d in snappergps_data])

###############################################################################
# Compare to ground truth
###############################################################################

# Arrays to store geodetic coordinates [decimal degrees]
lat = []
lon = []

idx = 0
# Read geodetic coordinates from file
with open(gt_file, "rb") as stream:
    ubr = UBXReader(stream, ubxonly=True)
    for (raw_data, parsed_data) in ubr:
        if parsed_data.identity == "NAV-PVT":
            if (parsed_data.min >= start_min or idx > 0) and idx < len(time):
                print(f"{parsed_data.min:02d}:{parsed_data.second:02d}")
                lat.append(parsed_data.lat*1e-7)  # Convert to degrees
                lon.append(parsed_data.lon*1e-7)  # Convert to degrees
                idx += 1

# Transform geodetic coordinates into east-north-up coordinates [m]
gt_e, gt_n, gt_u = pm.geodetic2enu(np.array(lat), np.array(lon),
                                   np.zeros(len(lat)), lat0, lon0, 0)

# Plot track
# Purple #5E2590 (94,37,144)
# Pink #B390CF (179,144,207)
# Bright grey #999999 (153,153,153)
# Dark grey #5C5C5C (92,92,92)
# Black #000000 (0,0,0)
# White #FFFFFF (255,255,255)
fig, ax = plt.subplots()
plt.plot(e, n, "x-", label="SnapperGPS", markersize=1, color="#B390CF",
         linewidth=0.5)
# plt.plot(x_post[:, 1], x_post[:, 2], "x", label="EKF", markersize=3)
plt.plot(x_smooth[:, 1], x_smooth[:, 2], "x-", label="RTS", markersize=1,
         color="#5C5C5C")
plt.plot(x_gpr[:, 0], x_gpr[:, 1], "x-", label="GPR", markersize=1,
         color="#5E2590")
plt.plot(gt_e, gt_n, "-", label="ground truth", linewidth=3, color="#000000")
x_lim = ax.get_xlim()
y_lim = ax.get_ylim()
ax.set_aspect('equal', adjustable='box')
plt.xlim(x_lim)
plt.ylim(y_lim)
plt.grid()
plt.xlabel("east [m]")
plt.ylabel("north [m]")
plt.title(snappergps_file.split(".")[0])
plt.legend()
plt.show()


def _error_stat(est, gt, method):
    """Mean and standard deviation of horizontal error."""
    err = np.linalg.norm(est - gt, axis=1)
    m = np.mean(err)
    sigma = np.std(err)
    median = np.median(err)
    print(f"{method:<10} | {m:>12.2f} ± {sigma:>7.2f} | {median:>16.2f}")


print()
print("method     | mean error ± stdev [m] | median error [m]")
print("-------------------------------------------------------")
gt = np.array([gt_e, gt_n]).T
_error_stat(x_post[:, 1:3], gt, "EKF")
_error_stat(x_smooth[:, 1:3], gt, "RTS")
_error_stat(x_gpr[:, :2], gt, "GPR")
_error_stat(z[:, 1:3], gt, "SnapperGPS")
