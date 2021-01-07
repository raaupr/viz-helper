"""Some stuff for playing with ellipsoids.
    From https://github.com/minillinim/ellipsoid/blob/master/ellipsoid.py"""

#!/usr/bin/python

from __future__ import division

import numpy as np
import plotly.graph_objs as go
import streamlit as st
from numpy import linalg
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# @st.cache
def get_min_vol_ellipse(P, tolerance=0.01):
    """Find the minimum volume ellipsoid which holds all the points

    Based on work by Nima Moshtagh
    http://www.mathworks.com/matlabcentral/fileexchange/9542
    and also by looking at:
    http://cctbx.sourceforge.net/current/python/scitbx.math.minimum_covering_ellipsoid.html
    Which is based on the first reference anyway!

    Here, P is a numpy array of N dimensional points like this:
    P = [[x,y,z,...], <-- one point per line
            [x,y,z,...],
            [x,y,z,...]]

    Returns:
    (center, radii, rotation)

    """
    (N, d) = np.shape(P)
    d = float(d)

    # Q will be our working array
    Q = np.vstack([np.copy(P.T), np.ones(N)])
    QT = Q.T

    # initializations
    err = 1.0 + tolerance
    u = (1.0 / N) * np.ones(N)

    # Khachiyan Algorithm
    while err > tolerance:
        V = np.dot(Q, np.dot(np.diag(u), QT))
        M = np.diag(
            np.dot(QT, np.dot(linalg.inv(V), Q))
        )  # M the diagonal vector of an NxN matrix
        j = np.argmax(M)
        maximum = M[j]
        step_size = (maximum - d - 1.0) / ((d + 1.0) * (maximum - 1.0))
        new_u = (1.0 - step_size) * u
        new_u[j] += step_size
        err = np.linalg.norm(new_u - u)
        u = new_u

    # center of the ellipse
    center = np.dot(P.T, u)

    # the A matrix for the ellipse
    A = (linalg.inv(np.dot(P.T, np.dot(np.diag(u), P)) - np.array([[a * b for b in center] for a in center])) / d)

    # Get the values we'd like to return
    U, s, rotation = linalg.svd(A)
    radii = 1.0 / np.sqrt(s)

    return (center, radii, rotation)


@st.cache
def get_ellipsoid_volume(radii):
    """Calculate the volume of the blob"""
    return 4.0 / 3.0 * np.pi * radii[0] * radii[1] * radii[2]


@st.cache(allow_output_mutation=True)
def plot_ellipsoid(
    P,
    center,
    radii,
    rotation,
    plot_axes=True,
    P_text = None,
    cage_color="yellow",
    surface_color="blue",
    marker_color="red",
    axes_color="black",
):
    """Plot an ellipsoid"""
    x = P[:,0]
    y = P[:,1]
    z = P[:, 2]

    u = np.linspace(0.0, 2.0 * np.pi, 100)
    v = np.linspace(0.0, np.pi, 100)

    # cartesian coordinates that correspond to the spherical angles:
    x2 = radii[0] * np.outer(np.cos(u), np.sin(v))
    y2 = radii[1] * np.outer(np.sin(u), np.sin(v))
    z2 = radii[2] * np.outer(np.ones_like(u), np.cos(v))
    # rotate accordingly
    for i in range(len(x2)):
        for j in range(len(x2)):
            [x2[i, j], y2[i, j], z2[i, j]] = (
                np.dot([x2[i, j], y2[i, j], z2[i, j]], rotation) + center
            )

    if P_text is None:
        P_text = list(range(len(x)))

    fig = go.Figure(
        data=[
            go.Surface(
                x=x2,
                y=y2,
                z=z2,
                opacity=0.3,
                showscale=False,
                colorscale=[[0, surface_color], [1, surface_color]],
            ),
            go.Scatter3d(
                x=x,
                y=y,
                z=z,
                mode="markers+text",
                marker=dict(size=3, opacity=1, color=marker_color),
                text=P_text
            ),
            go.Scatter3d(
                x=x2.flatten(),
                y=y2.flatten(),
                z=z2.flatten(),
                mode="lines",
                marker=dict(size=1, opacity=0.9, color=cage_color),
            ),
        ]
    )

    if plot_axes:
        # make some purdy axes
        axes = np.array(
            [[radii[0], 0.0, 0.0], [0.0, radii[1], 0.0], [0.0, 0.0, radii[2]]]
        )
        # rotate accordingly
        for i in range(len(axes)):
            axes[i] = np.dot(axes[i], rotation)
        # plot axes
        for p in axes:
            X3 = np.linspace(-p[0], p[0], 100) + center[0]
            Y3 = np.linspace(-p[1], p[1], 100) + center[1]
            Z3 = np.linspace(-p[2], p[2], 100) + center[2]
            fig.add_trace(
                go.Scatter3d(
                    x=X3, y=Y3, z=Z3, marker=dict(size=1, opacity=0.9, color=axes_color)
                )
            )

    minx = min(min(x), min(x2.flatten()))
    miny = min(min(y), min(y2.flatten()))
    minz = min(min(z), min(z2.flatten()))
    maxx = max(max(x), max(x2.flatten()))
    maxy = max(max(y), max(y2.flatten()))
    maxz = max(max(z), max(z2.flatten()))

    fig.update_traces(showlegend=False)
    return fig, (minx, miny, minz, maxx, maxy, maxz)

@st.cache(allow_output_mutation=True)
def plot_ellipsoid_matplotlib(
    fig,
    P,
    center,
    radii,
    rotation,
    plot_axes=True,
    P_text = None,
    cage_color="yellow",
    surface_color="blue",
    marker_color="red",
    axes_color="black",
):
    ax = Axes3D(fig)

    # -- plot points
    ax.scatter(P[:,0], P[:,1], P[:,2], color=marker_color, marker='o', s=100)

    # -- plot sphere
    u = np.linspace(0.0, 2.0 * np.pi, 100)
    v = np.linspace(0.0, np.pi, 100)

    # cartesian coordinates that correspond to the spherical angles:
    x = radii[0] * np.outer(np.cos(u), np.sin(v))
    y = radii[1] * np.outer(np.sin(u), np.sin(v))
    z = radii[2] * np.outer(np.ones_like(u), np.cos(v))
    # rotate accordingly
    for i in range(len(x)):
        for j in range(len(x)):
            [x[i,j],y[i,j],z[i,j]] = np.dot([x[i,j],y[i,j],z[i,j]], rotation) + center

    if plot_axes:
        # make some purdy axes
        axes = np.array([[radii[0],0.0,0.0],
                        [0.0,radii[1],0.0],
                        [0.0,0.0,radii[2]]])
        # rotate accordingly
        for i in range(len(axes)):
            axes[i] = np.dot(axes[i], rotation)

        # plot axes
        for p in axes:
            X3 = np.linspace(-p[0], p[0], 100) + center[0]
            Y3 = np.linspace(-p[1], p[1], 100) + center[1]
            Z3 = np.linspace(-p[2], p[2], 100) + center[2]
            ax.plot(X3, Y3, Z3, color=axes_color)

    # plot ellipsoid
    ax.plot_wireframe(x, y, z,  rstride=4, cstride=4, color=cage_color, alpha=0.2)
            
    # ax.azim += 1    
    # elev = 0.    
    # for ii in range(0,360,1):
    #     ax.view_init(elev=elev, azim=ii)
    #         savefig("movie%d.png" % ii)
    # print(ax.azim, ax.elev, ax.dist)

    # ax.elev = 30. # angle between the eye and the xy plane.
    # ax.azim = -60 # rotation around the z axis; 0 means "looking from +x", 90 means "looking from +y"
    # ax.dist = 20 # distance from the center visible point in data coordinates.

    minx, maxx = ax.get_xbound()
    miny, maxy = ax.get_ybound()
    minz, maxz = ax.get_zbound()

    return fig, (minx, miny, minz, maxx, maxy, maxz)