from scipy.spatial import Delaunay
import numpy as np
from collections import defaultdict
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from queue import Queue
import streamlit as st


@st.cache
def alpha_shape_3d_autoalpha(pos):
    """
    Based on: https://stackoverflow.com/questions/26303878/alpha-shapes-in-3d
    Compute the alpha shape (concave hull) of a set of 3D points.
    Alpha value is computed automatically to include all points as vertices.
    Parameters:
        pos - np.array of shape (n,3) points.
    return
        outer surface vertex indices, edge indices, and triangle indices
    """

    tetra = Delaunay(pos)
    # Find radius of the circumsphere.
    # By definition, radius of the sphere fitting inside the tetrahedral needs
    # to be smaller than alpha value
    # http://mathworld.wolfram.com/Circumsphere.html
    tetrapos = np.take(pos, tetra.vertices, axis=0)
    normsq = np.sum(tetrapos ** 2, axis=2)[:, :, None]
    ones = np.ones((tetrapos.shape[0], tetrapos.shape[1], 1))
    a = np.linalg.det(np.concatenate((tetrapos, ones), axis=2))
    Dx = np.linalg.det(np.concatenate((normsq, tetrapos[:, :, [1, 2]], ones), axis=2))
    Dy = -np.linalg.det(np.concatenate((normsq, tetrapos[:, :, [0, 2]], ones), axis=2))
    Dz = np.linalg.det(np.concatenate((normsq, tetrapos[:, :, [0, 1]], ones), axis=2))
    c = np.linalg.det(np.concatenate((normsq, tetrapos), axis=2))
    r = np.sqrt(Dx ** 2 + Dy ** 2 + Dz ** 2 - 4 * a * c) / (2 * np.abs(a))

    sorted_r = sorted(list(set(r)))[::-1]

    for i in range(len(sorted_r)):
        if i == len(sorted_r) - 1:
            diff = 0.1
        else:
            diff = (sorted_r[i + 1] - sorted_r[i]) / 2
        alpha = sorted_r[i] + diff
        # Find tetrahedrals
        tetras = tetra.vertices[r < alpha, :]
        # triangles
        TriComb = np.array([(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)])
        Triangles = tetras[:, TriComb].reshape(-1, 3)
        Triangles = np.sort(Triangles, axis=1)
        # Remove triangles that occurs twice, because they are within shapes
        TrianglesDict = defaultdict(int)
        for tri in Triangles:
            TrianglesDict[tuple(tri)] += 1
        Triangles = np.array([tri for tri in TrianglesDict if TrianglesDict[tri] == 1])
        # edges
        EdgeComb = np.array([(0, 1), (0, 2), (1, 2)])
        Edges = Triangles[:, EdgeComb].reshape(-1, 2)
        Edges = np.sort(Edges, axis=1)
        Edges = np.unique(Edges, axis=0)
        Vertices = np.unique(Edges)
        if len(Vertices) == len(pos):
            break

    return Vertices, Edges, Triangles, tetras


@st.cache(allow_output_mutation=True)
def plot_alphashape(pts, triangles, text=None, compute_range=False):
    if text is None:
        text = [str(i) for i in range(len(pts))]
    fig = make_subplots()
    minx = miny = minz = 999999999
    maxx = maxy = maxz = -999999999
    for s in triangles:
        s = np.append(s, s[0])
        vtx = np.array([pts[i] for i in s])
        trace = go.Mesh3d(
            x=vtx[:, 0],
            y=vtx[:, 1],
            z=vtx[:, 2],
            i=[0, 0, 0, 1],
            j=[1, 2, 3, 2],
            k=[2, 3, 1, 3],
            name="y",
        )
        fig.add_trace(trace)
        if compute_range:
            minx = min(minx, min(vtx[:, 0]))
            miny = min(miny, min(vtx[:, 1]))
            minz = min(minz, min(vtx[:, 2]))
            maxx = max(maxx, max(vtx[:, 0]))
            maxy = max(maxy, max(vtx[:, 1]))
            maxz = max(maxz, max(vtx[:, 2]))
    fig.add_trace(
        go.Scatter3d(
            x=pts[:, 0], y=pts[:, 1], z=pts[:, 2], mode="markers+text", text=text
        )
    )
    if compute_range:
        return fig, (minx, miny, minz, maxx, maxy, maxz)
    return fig


@st.cache
def switch(cur_tri, tri, same_elmts):
    idx0_i = np.where(cur_tri == same_elmts[0])[0][0]
    idx1_i = np.where(cur_tri == same_elmts[1])[0][0]
    idx_other_i = np.where((cur_tri != same_elmts[0]) & (cur_tri != same_elmts[1]))[0][
        0
    ]
    idx_other_j = np.where((tri != same_elmts[0]) & (tri != same_elmts[1]))[0][0]
    other_j = tri[idx_other_j]
    new_triangle = [-1, -1, -1]
    new_triangle[idx0_i] = same_elmts[1]
    new_triangle[idx1_i] = same_elmts[0]
    new_triangle[idx_other_i] = other_j
    return new_triangle


@st.cache
def orient_faces(triangles):
    # -- prepare queues
    next_tri = Queue()
    remaining_tri = Queue()
    for tri in triangles:
        remaining_tri.put(tri)
    # -- process triangles
    res = []
    while not remaining_tri.empty() or not next_tri.empty():
        if next_tri.empty():
            next_tri.put(remaining_tri.get())
        new_remain = Queue()
        cur_tri = next_tri.get()
        res.append(cur_tri)
        while not remaining_tri.empty():
            tri = remaining_tri.get()
            same_elmts = list(set(cur_tri) & set(tri))
            if len(same_elmts) < 2:
                new_remain.put(tri)
            else:
                new_tri = switch(cur_tri, tri, same_elmts)
                next_tri.put(new_tri)
        remaining_tri = new_remain
    return res


@st.cache
def compute_volume(pts, triangles):
    """https://math.stackexchange.com/questions/803076/how-to-calculate-volume-of-non-convex-polyhedron"""
    triangles = orient_faces(triangles)
    volume = 0
    for tri in triangles:
        face = pts[tri].transpose()
        volume += np.linalg.det(face) / 6
    volume = abs(volume)
    return volume


def compute_volume_tetras(pts, triangles, tetras):
    volume = 0
    for plane in triangles:
        plane_coords = pts[plane]
        for tetra in tetras:
            same_elmts = list(set(plane) & set(tetra))
            if len(same_elmts) == 3:
                break
        idx_other = np.where(
            (tetra != same_elmts[0])
            & (tetra != same_elmts[1])
            & (tetra != same_elmts[2])
        )[0][0]
        idx_other = tetra[idx_other]
        A, B, C = plane_coords
        X = pts[idx_other]
        Bp = B - A
        Cp = C - A
        Xp = X - A
        plane_sign = np.linalg.det(np.array([Bp, Cp, Xp]))
        plane_vol = np.linalg.det(plane_coords) / 6
        if plane_sign > 0:
            sign = 1
        else:
            sign = -1
        volume += sign * plane_vol
    return abs(volume)