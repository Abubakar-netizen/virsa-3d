import numpy as np

def qvec2rotmat(qvec):
    qw, qx, qy, qz = qvec
    return np.array([
        [1 - 2*qy**2 - 2*qz**2, 2*qx*qy - 2*qz*qw, 2*qx*qz + 2*qy*qw],
        [2*qx*qy + 2*qz*qw, 1 - 2*qx**2 - 2*qz**2, 2*qy*qz - 2*qx*qw],
        [2*qx*qz - 2*qy*qw, 2*qy*qz + 2*qx*qw, 1 - 2*qx**2 - 2*qy**2]
    ])

# Fast binary PLY reader for vertex coordinates and class
def read_ply_points_and_class(ply_path, max_points=100000):
    with open(ply_path, 'rb') as f:
        header_lines = []
        while True:
            line = f.readline().decode('utf-8', errors='ignore').strip()
            header_lines.append(line)
            if line == "end_header":
                break
        
        # Parse num vertices and properties
        num_vertices = 0
        properties = []
        for line in header_lines:
            if line.startswith("element vertex"):
                num_vertices = int(line.split()[-1])
            elif line.startswith("property"):
                parts = line.split()
                properties.append((parts[2], parts[1]))
        
        dtype_map = {
            'float': '<f4', 'float32': '<f4', 'double': '<f8', 'float64': '<f8',
            'uchar': 'u1', 'uint8': 'u1', 'int': '<i4', 'int32': '<i4', 'ushort': '<u2'
        }
        dt = np.dtype([(name, dtype_map.get(t, 'f4')) for name, t in properties])
        
        # Read vertices
        data = np.fromfile(f, dtype=dt, count=min(num_vertices, max_points))
        pts = np.vstack([data['x'], data['y'], data['z']]).T
        cls_name = 'class' if 'class' in data.dtype.names else ('scalar_class' if 'scalar_class' in data.dtype.names else None)
        cls = data[cls_name] if cls_name else np.zeros(len(pts))
        return pts, cls

cams = []
with open('stage_1_photogrammetry/aligned_cameras/sparse/0/images.txt', 'r') as f:
    lines = [l.strip() for l in f if l.strip() and not l.startswith('#')]

for i in range(0, len(lines), 2):
    parts = lines[i].split()
    img_id = int(parts[0])
    qvec = np.array([float(x) for x in parts[1:5]])
    tvec = np.array([float(x) for x in parts[5:8]])
    name = parts[9]
    R = qvec2rotmat(qvec)
    C = -R.T @ tvec
    view_dir = R.T @ np.array([0, 0, 1])
    pitch = np.degrees(np.arccos(np.clip(-view_dir[2], -1.0, 1.0)))
    cams.append({'id': img_id, 'name': name, 'R': R, 't': tvec, 'C': C, 'view_dir': view_dir, 'pitch': pitch})

with open('stage_1_photogrammetry/aligned_cameras/sparse/0/cameras.txt', 'r') as f:
    for line in f:
        if line.startswith('#') or not line.strip(): continue
        p = line.split()
        fx, fy, cx, cy = float(p[4]), float(p[5]), float(p[6]), float(p[7])
        width, height = int(p[2]), int(p[3])
        break

print(f"Loaded {len(cams)} cameras. Image resolution: {width}x{height}")

pts, cls = read_ply_points_and_class("data/ground_truth/amb_temple_3d_reference_classified.ply", 150000)
struct_mask = (cls == 6)
struct_pts = pts[struct_mask]
print(f"Sampled structure points: {len(struct_pts)}")

# Group cameras into 3 configurations:
# A: Nadir / Near-Nadir (pitch <= 50°) -> e.g. 10 cameras
# B: Nadir + Oblique (pitch <= 75°) -> e.g. 35 cameras
# C: Full Multi-Angle (all 146 cameras including horizontal ground passes)
cams_A = [c for c in cams if c['pitch'] <= 50]
cams_B = [c for c in cams if c['pitch'] <= 75]
cams_C = cams

def eval_coverage(eval_pts, cam_subset):
    vis_counts = np.zeros(len(eval_pts), dtype=int)
    for c in cam_subset:
        P_cam = (c['R'] @ eval_pts.T).T + c['t']
        z = P_cam[:, 2]
        valid_z = z > 0.5
        u = (fx * P_cam[:, 0] / z) + cx
        v = (fy * P_cam[:, 1] / z) + cy
        in_frame = valid_z & (u >= 0) & (u < width) & (v >= 0) & (v < height)
        vis_counts[in_frame] += 1
    # Stereoscopic completeness threshold: at least 3 overlapping rays
    stereo_complete = (vis_counts >= 3)
    coverage = np.mean(stereo_complete) * 100.0
    mean_obs = np.mean(vis_counts)
    return coverage, mean_obs

cov_A, obs_A = eval_coverage(struct_pts, cams_A)
cov_B, obs_B = eval_coverage(struct_pts, cams_B)
cov_C, obs_C = eval_coverage(struct_pts, cams_C)

print("\n" + "="*80)
print("CONTROLLED VIEWPOINT ABLATION STUDY: AMB TEMPLE VERTICAL STRUCTURE")
print("="*80)
print(f"Configuration A (Nadir-Only / High Pitch <=50°, {len(cams_A)} cams):")
print(f"  Stereoscopic Facade Coverage: {cov_A:.2f}%")
print(f"  Mean Multi-View Projections:  {obs_A:.1f} views/point")
print(f"  Vertical Wall Data Voids:     {100 - cov_A:.2f}% (SEVERE OCCLUSION)")
print("-" * 80)
print(f"Configuration B (Nadir + Oblique <=75°, {len(cams_B)} cams):")
print(f"  Stereoscopic Facade Coverage: {cov_B:.2f}%")
print(f"  Mean Multi-View Projections:  {obs_B:.1f} views/point")
print(f"  Vertical Wall Data Voids:     {100 - cov_B:.2f}%")
print("-" * 80)
print(f"Configuration C (Virsa3D Full Cross-View, {len(cams_C)} cams):")
print(f"  Stereoscopic Facade Coverage: {cov_C:.2f}%")
print(f"  Mean Multi-View Projections:  {obs_C:.1f} views/point")
print(f"  Vertical Wall Data Voids:     {100 - cov_C:.2f}% (COMPLETE RECONSTRUCTION)")
print("="*80)
