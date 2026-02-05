import numpy as np

def nodes_to_waypoints(path_nodes):
    """
    path_nodes: list of Node objects, each node.position is [x,y,z]
    returns (M,3) array
    """
    return np.array([node.position[0:3] for node in path_nodes])


def yaw_from_waypoints(ref_seq):
    # ref_seq: (N+1,3)
    dx = ref_seq[1,0] - ref_seq[0,0]
    dy = ref_seq[1,1] - ref_seq[0,1]
    return np.arctan2(dy, dx)


def resample_polyline(pts, ds):
    """
    Resample polyline with approx constant arc-length spacing ds.
    pts: (M,3), an ordered list of 3D points
    returns (R,3)

    Global planners produce unevenly spaced waypoints.
    MPPI/MPC assumes roughly constant spatial progress per timestep (needs uniform spacing).
    Therefore, we must resample the path so that consecutive reference
    points are evenly spaced in distance, not in index.
    """

    if len(pts) < 2:
        return pts.copy()

    # compute distances between consecutive points
    seg_len = []
    for i in range(len(pts) - 1):
        d = np.linalg.norm(pts[i+1] - pts[i])
        seg_len.append(d)

    seg_len = np.array(seg_len)

    # cumulative distance along the path
    s = np.zeros(len(pts))
    for i in range(1, len(pts)):
        s[i] = s[i-1] + seg_len[i-1]

    total_length = s[-1]
    if total_length < 1e-9:
        return pts[[0]].copy()

    # new equally spaced distances
    s_new = np.arange(0.0, total_length + ds, ds)

    # interpolate x, y, z independently
    out = np.zeros((len(s_new), 3))
    for j in range(3):
        out[:, j] = np.interp(s_new, s, pts[:, j])

    return out


def make_ref_sequence(waypoints, start_index, N):
    """
    Return ref positions over horizon:
    ref_seq[t] = [x_ref(t), y_ref(t), z_ref(t)]
    ref_seq[t] = waypoint[start_index + t]
    """
    M = len(waypoints)
    ref = np.zeros((N + 1, 3))

    for t in range(N + 1):
        idx = min(start_index + t, M - 1)
        ref[t] = waypoints[idx]

    return ref
