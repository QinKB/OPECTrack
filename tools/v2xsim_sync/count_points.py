"""Compute ground-truth box geometry and LiDAR point counts for both agents."""
import json
import os
import pickle
import sys

import numpy as np
from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.utils.geometry_utils import points_in_box, transform_matrix
from pyquaternion import Quaternion

from tools.nuscenes_tables import NuScenesTables

root, defects_path, out = sys.argv[1:4]
defects = set(json.load(open(defects_path)))
nusc = NuScenesTables(version='v1.0-trainval', dataroot=root, verbose=False)


def s2g(sd):
    cs = nusc.get('calibrated_sensor', sd['calibrated_sensor_token']); ep = nusc.get('ego_pose', sd['ego_pose_token'])
    return (transform_matrix(ep['translation'], Quaternion(ep['rotation'])) @
            transform_matrix(cs['translation'], Quaternion(cs['rotation']))), ep, cs


res = {}
for sample in nusc.sample:
    sd1 = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
    sd0 = nusc.get('sample_data', sample['data']['LIDAR_TOP_id_0'])
    T1, ep1, cs1 = s2g(sd1)
    T0, _, _ = s2g(sd0)
    T01 = np.linalg.inv(T1) @ T0
    T10 = np.linalg.inv(T01)
    p1 = LidarPointCloud.from_file(os.path.join(root, sd1['filename'])).points[:3]
    p0 = LidarPointCloud.from_file(os.path.join(root, sd0['filename'])).points[:3]
    p0 = T01[:3, :3] @ p0 + T01[:3, 3:4]
    _, boxes, _ = nusc.get_sample_data(sd1['token'])
    n = len(sample['anns'])
    assert len(boxes) == n
    l2e = np.asarray(cs1['translation'])
    assert np.allclose(Quaternion(cs1['rotation']).rotation_matrix, np.eye(3), atol=1e-6), 'lidar2ego rotation'
    f = dict(category=[], d_ego=np.zeros(n), d_lidar=np.zeros(n), d_rsu=np.zeros(n), rsu_x=np.zeros(n),
             rsu_y=np.zeros(n), is_ego=np.zeros(n, bool), size_defect=np.zeros(n, bool),
             npts_veh=np.full(n, -1, np.int32), npts_inf=np.full(n, -1, np.int32))
    for i, (tok, box) in enumerate(zip(sample['anns'], boxes)):
        ann = nusc.get('sample_annotation', tok)
        assert box.token == tok
        c = box.center
        f['category'].append(ann['category_name'])
        f['d_lidar'][i] = np.hypot(c[0], c[1])
        f['d_ego'][i] = np.hypot(c[0] + l2e[0], c[1] + l2e[1])
        cr = T10[:3, :3] @ c + T10[:3, 3]
        f['rsu_x'][i], f['rsu_y'][i] = cr[0], cr[1]
        f['d_rsu'][i] = np.hypot(cr[0], cr[1])
        f['is_ego'][i] = np.hypot(ann['translation'][0] - ep1['translation'][0],
                                  ann['translation'][1] - ep1['translation'][1]) < 0.01
        f['size_defect'][i] = tok in defects
        if f['d_lidar'][i] <= 60.0:
            f['npts_veh'][i] = int(points_in_box(box, p1, wlh_factor=1.0).sum())
        if f['d_rsu'][i] <= 75.0:
            f['npts_inf'][i] = int(points_in_box(box, p0, wlh_factor=1.0).sum())
    f['category'] = np.array(f['category'])
    res[sample['token']] = f
with open(out, 'wb') as stream:
    pickle.dump(res, stream)
