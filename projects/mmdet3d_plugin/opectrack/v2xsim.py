"""V2X-Sim adapters of the V2X-Seq-SPD dataset classes used by OPECTrack (detectors and tracker).

The V2X-Sim infos written by tools/v2xsim_sync/make_sync_data.py have the SPD shape (ground-truth arrays already
restricted to the protocol's ground truth, valid_flag all True, timestamp_inf = timestamp, token_inf = token).
What differs from SPD and is changed here, nothing else:
  * point-cloud paths: SPD builds ``velodyne/{int(token[1:]):06d}.pcd`` from its numeric frame ids; V2X-Sim tokens are
    random strings, the LiDAR file is the one in ``lidar_path`` (``./datasets/<root>/sweeps/LIDAR_TOP_id_k/...``,
    relative part ``sweeps/LIDAR_TOP_id_k/...``), the RSU file of a cooperative
    frame is the same file name under ``LIDAR_TOP_id_0`` (LET-VIC's rule);
  * the files are nuScenes ``.pcd.bin`` (5 float32 per point), read with ``XETVICV2XSimLoadMultiViewPointsFromFile``
    in the configs (pipeline), not here.
"""
import numpy as np
from mmdet.datasets import DATASETS
from nuscenes.eval.common.utils import Quaternion, quaternion_yaw

from .mm.spd_det_dataset import SPDClipDataset, SPDDetDataset


def _rel(path):
    """``./datasets/<root>/sweeps/LIDAR_TOP_id_k/file`` -> ``sweeps/LIDAR_TOP_id_k/file``."""
    parts = path.split('/')
    k = parts.index('sweeps')
    return '/'.join(parts[k:])


@DATASETS.register_module()
class V2XSimDetDataset(SPDDetDataset):

    def get_data_info(self, index):
        info = self.data_infos[index]
        input_dict = dict(
            sample_idx=info["token"], pts_filename=info["lidar_path"], sweeps=info["sweeps"],
            scene_token=info["scene_token"], frame_idx=info["frame_idx"], timestamp=info["timestamp"] / 1e6,
            pc_filename=[_rel(info["lidar_path"])],
            pc_range_veh=self.pc_range_veh, pc_range_inf=self.pc_range_inf)
        input_dict["ann_info"] = self.get_ann_info(index)
        return input_dict


@DATASETS.register_module()
class V2XSimClipDataset(SPDClipDataset):
    """``SPDClipDataset.get_data_info`` with the V2X-Sim point-cloud paths; everything else is that method."""

    def get_data_info(self, index):
        if self.modality["use_camera"]:
            raise NotImplementedError("LiDAR only")
        info = self.data_infos[index]
        token_inf = info["token_inf"] if "token_inf" in info else -1
        input_dict = dict(
            sample_idx=info["token"], sample_idx_inf=token_inf, pts_filename=info["lidar_path"],
            sweeps=info["sweeps"], ego2global_translation=info["ego2global_translation"],
            ego2global_rotation=info["ego2global_rotation"], prev_idx=info["prev"], next_idx=info["next"],
            scene_token=info["scene_token"], can_bus=info["can_bus"], frame_idx=info["frame_idx"],
            timestamp=info["timestamp"] / 1e6, pc_range_veh=self.pc_range_veh, pc_range_inf=self.pc_range_inf)
        l2e_r_mat = Quaternion(info["lidar2ego_rotation"]).rotation_matrix
        e2g_r_mat = Quaternion(info["ego2global_rotation"]).rotation_matrix
        l2g_r_mat = l2e_r_mat.T @ e2g_r_mat.T
        l2g_t = info["lidar2ego_translation"] @ e2g_r_mat.T + info["ego2global_translation"]
        input_dict.update(l2g_r_mat=l2g_r_mat.astype(np.float32), l2g_t=l2g_t.astype(np.float32))
        veh2inf_rt = np.eye(4)
        if "VehLidar2InfLidar_rotation" in info and "VehLidar2InfLidar_translation" in info:
            veh2inf_rt[:3, :3] = info["VehLidar2InfLidar_rotation"]
            veh2inf_rt[:3, 3] = info["VehLidar2InfLidar_translation"]
            veh2inf_rt = veh2inf_rt.T
        input_dict["veh2inf_rt"] = veh2inf_rt.astype(np.float32)
        timestamps = [info["timestamp"]]
        if "token_inf" in info:
            timestamps.append(info["timestamp_inf"])
        input_dict["timestamps"] = timestamps

        def view(pc_range, lidar2lidar):
            pad = np.eye(4)
            pad[0, 3] -= pc_range[0]
            pad[1, 3] -= pc_range[1]
            pad[2, 2] = 0
            pad[2, 3] = 1
            return pad, pad @ lidar2lidar

        own = _rel(info["lidar_path"])
        paths = [own]
        pad, proj = view(self.pc_range_veh, np.eye(4))
        intrinsics, projections, lidar2lidar = [pad], [proj], [np.eye(4)]
        if "token_inf" in info:
            paths.append(own.replace("LIDAR_TOP_id_1", "LIDAR_TOP_id_0"))
            veh2inf = input_dict["veh2inf_rt"].T
            pad, proj = view(self.pc_range_inf, veh2inf)
            intrinsics.append(pad)
            projections.append(proj)
            lidar2lidar.append(veh2inf)
        input_dict.update(pc_filename=paths, lidar2pseudoimg=projections, lidar_intrinsic=intrinsics,
                          lidar2lidar=lidar2lidar)
        input_dict["ann_info"] = self.get_ann_info(index)
        rotation = Quaternion(input_dict["ego2global_rotation"])
        can_bus = input_dict["can_bus"]
        can_bus[:3] = input_dict["ego2global_translation"]
        can_bus[3:7] = rotation
        angle = quaternion_yaw(rotation) / np.pi * 180
        if angle < 0:
            angle += 360
        can_bus[-2] = angle / 180 * np.pi
        can_bus[-1] = angle
        input_dict["can_bus"] = can_bus
        return input_dict
