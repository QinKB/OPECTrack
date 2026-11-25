import mmcv
import numpy as np
import os
import json
from nuscenes.prediction import PredictHelper
from os import path as osp
from pyquaternion import Quaternion
from mmdet3d.datasets import NuScenesDataset
from tools.nuscenes_tables import NuScenesTables

save_valid_flag_record_path = './datasets/V2X-Sim-full-coop/valid_flag_record.json'


load_valid_flag_record_path = './datasets/V2X-Sim-full-id0/valid_flag_record.json'


def load_json(path):
    with open(path, mode="r") as f:
        data = json.load(f)
    return data


def write_json(data, path):
    with open(path, mode="w") as f:
        json.dump(data, f, indent=2)


def valid_points_yx(locs):
    valid_point_flags = []
    for loc in locs:
        x = loc[1]
        y = loc[0]
        valid_point_flags.append(y - x >= 0 and y + x >= 0 and y <= 50 and y >= 0)

    valid_point_flags = np.array(valid_point_flags, dtype=bool).reshape(-1)

    return valid_point_flags


def valid_points_yx_square(locs):
    valid_point_flags = []
    for loc in locs:
        x = loc[1]
        y = loc[0]
        valid_point_flags.append(x >= -50 and x <= 50 and y >= -50 and y <= 50)

    valid_point_flags = np.array(valid_point_flags, dtype=bool).reshape(-1)

    return valid_point_flags


def create_nuscenes_infos(root_path,
                          out_path,
                          can_bus_root_path,
                          info_prefix,
                          version='v1.0-trainval',
                          max_sweeps=10,
                          use_can_bus=True,
                          mono=False,
                          coop=False):
    """Create info file of nuscene dataset.

    Given the raw data, generate its related info file in pkl format.

    Args:
        root_path (str): Path of the data root.
        info_prefix (str): Prefix of the info file to be generated.
        version (str): Version of the data.
            Default: 'v1.0-trainval'
        max_sweeps (int): Max number of sweeps.
            Default: 10
    """
    from nuscenes.can_bus.can_bus_api import NuScenesCanBus
    print(version, root_path)
    nusc = NuScenesTables(version=version, dataroot=root_path, verbose=True)
    nusc_can_bus = NuScenesCanBus(dataroot=can_bus_root_path)
    splits_path = './data/split_datas/v2x-sim-split.json'
    splits = load_json(splits_path)['batch_split']
    train_scenes = splits['train']
    val_scenes = splits['val']


    # filter existing scenes.
    available_scenes = get_available_scenes(nusc)
    available_scene_names = [s['name'] for s in available_scenes]
    train_scenes = list(filter(lambda x: x in available_scene_names, train_scenes))
    val_scenes = list(filter(lambda x: x in available_scene_names, val_scenes))
    train_scenes = set([
        available_scenes[available_scene_names.index(s)]['token']
        for s in train_scenes
    ])
    val_scenes = set([
        available_scenes[available_scene_names.index(s)]['token']
        for s in val_scenes
    ])

    test = 'test' in version
    if test:
        print('test scene: {}'.format(len(train_scenes)))
    else:
        print('train scene: {}, val scene: {}'.format(
            len(train_scenes), len(val_scenes)))

    train_nusc_infos, val_nusc_infos = _fill_trainval_infos(
        nusc, nusc_can_bus, train_scenes, val_scenes, test, max_sweeps=max_sweeps,use_can_bus=use_can_bus,mono=mono, coop=coop)

    metadata = dict(version=version)
    if test:
        print('test sample: {}'.format(len(train_nusc_infos)))
        data = dict(infos=train_nusc_infos, metadata=metadata)
        info_path = osp.join(out_path,
                             '{}_infos_temporal_test.pkl'.format(info_prefix))
        mmcv.dump(data, info_path)
    else:
        print('train sample: {}, val sample: {}'.format(
            len(train_nusc_infos), len(val_nusc_infos)))
        data = dict(infos=train_nusc_infos, metadata=metadata)
        info_path = osp.join(out_path,
                             '{}_infos_temporal_train.pkl'.format(info_prefix))
        mmcv.dump(data, info_path)
        data['infos'] = val_nusc_infos
        info_val_path = osp.join(out_path,
                                 '{}_infos_temporal_val.pkl'.format(info_prefix))
        mmcv.dump(data, info_val_path)


def get_available_scenes(nusc):
    """Get available scenes from the input nuscenes class.

    Given the raw data, get the information of available scenes for
    further info generation.

    Args:
        nusc (class): Dataset class in the nuScenes dataset.

    Returns:
        available_scenes (list[dict]): List of basic information for the
            available scenes.
    """
    available_scenes = []
    print('total scene num: {}'.format(len(nusc.scene)))
    for scene in nusc.scene:
        scene_token = scene['token']
        scene_rec = nusc.get('scene', scene_token)
        sample_rec = nusc.get('sample', scene_rec['first_sample_token'])
        sd_rec = nusc.get('sample_data', sample_rec['data']['LIDAR_TOP'])
        has_more_frames = True
        scene_not_exist = False
        while has_more_frames:
            lidar_path, boxes, _ = nusc.get_sample_data(sd_rec['token'])
            lidar_path = str(lidar_path)
            if os.getcwd() in lidar_path:
                # path from lyftdataset is absolute path
                lidar_path = lidar_path.split(f'{os.getcwd()}/')[-1]
                # relative path
            if not mmcv.is_filepath(lidar_path):
                scene_not_exist = True
                break
            else:
                break
        if scene_not_exist:
            continue
        available_scenes.append(scene)
    print('exist scene num: {}'.format(len(available_scenes)))
    return available_scenes


def _get_can_bus_info(nusc, nusc_can_bus, sample):
    scene_name = nusc.get('scene', sample['scene_token'])['name']
    sample_timestamp = sample['timestamp']
    try:
        pose_list = nusc_can_bus.get_messages(scene_name, 'pose')
    except:
        return np.zeros(18)  # server scenes do not have can bus information.
    can_bus = []
    # during each scene, the first timestamp of can_bus may be large than the first sample's timestamp
    last_pose = pose_list[0]
    for i, pose in enumerate(pose_list):
        if pose['utime'] > sample_timestamp:
            break
        last_pose = pose
    _ = last_pose.pop('utime')  # useless
    pos = last_pose.pop('pos')
    rotation = last_pose.pop('orientation')
    can_bus.extend(pos)
    can_bus.extend(rotation)
    for key in last_pose.keys():
        can_bus.extend(pose[key])  # 16 elements
    can_bus.extend([0., 0.])
    return np.array(can_bus)


def _get_future_traj_info(nusc, sample, predict_steps=16):
    sample_token = sample['token']
    ann_tokens = np.array(sample['anns'])
    fut_traj_all = []
    fut_traj_valid_mask_all = []
    predict_helper = PredictHelper(nusc)
    for ann_token in ann_tokens:
        instance_token = nusc.get('sample_annotation', ann_token)['instance_token']
        fut_traj_local = predict_helper.get_future_for_agent(instance_token,
                                                            sample_token,
                                                            seconds=predict_steps//2,
                                                            in_agent_frame=True)

        fut_traj = np.zeros((predict_steps, 2))
        fut_traj_valid_mask = np.zeros((predict_steps, 2))
        if fut_traj_local.shape[0] > 0:
            length = fut_traj_local.shape[0]
            fut_traj[:length] = fut_traj_local
            fut_traj_valid_mask[:length] = 1
        fut_traj_all.append(fut_traj)
        fut_traj_valid_mask_all.append(fut_traj_valid_mask)
    if len(ann_tokens) > 0:
        fut_traj_all = np.stack(fut_traj_all, axis=0)
        fut_traj_valid_mask_all = np.stack(fut_traj_valid_mask_all, axis=0)
    else:
        fut_traj_all = np.zeros((0, predict_steps, 2))
        fut_traj_valid_mask_all = np.zeros((0, predict_steps, 2))

    return fut_traj_all, fut_traj_valid_mask_all


def _fill_trainval_infos(nusc,
                         nusc_can_bus,
                         train_scenes,
                         val_scenes,
                         test=False,
                         max_sweeps=10,
                         use_can_bus=True,
                         mono=False,
                         coop=False):
    """Generate the train/val infos from the raw data.

    Args:
        nusc (:obj:`NuScenes`): Dataset class in the nuScenes dataset.
        train_scenes (list[str]): Basic information of training scenes.
        val_scenes (list[str]): Basic information of validation scenes.
        test (bool): Whether use the test mode. In the test mode, no
            annotations can be accessed. Default: False.
        max_sweeps (int): Max number of sweeps. Default: 10.

    Returns:
        tuple[list[dict]]: Information of training set and validation set
            that will be saved to the info file.
    """
    train_nusc_infos = []
    val_nusc_infos = []
    frame_idx = 0
    valid_flag_record = {}
    if coop:
        cav_valid_flag_record = load_json(load_valid_flag_record_path)

    for sample in mmcv.track_iter_progress(nusc.sample):
        lidar_token = sample['data']['LIDAR_TOP']
        sd_rec = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
        cs_record = nusc.get('calibrated_sensor', sd_rec['calibrated_sensor_token'])
        pose_record = nusc.get('ego_pose', sd_rec['ego_pose_token'])
        lidar_path, boxes, _ = nusc.get_sample_data(lidar_token)

        # mmcv.check_file_exist(lidar_path)
        can_bus = _get_can_bus_info(nusc, nusc_can_bus, sample)
        if not use_can_bus:
            can_bus = np.zeros(18,dtype=float)
        ##
        info = {
            'lidar_path': lidar_path,
            'token': sample['token'],
            'prev': sample['prev'],
            'next': sample['next'],
            'can_bus': can_bus,
            'frame_idx': frame_idx,  # temporal related info
            'sweeps': [],
            'cams': dict(),
            'scene_token': sample['scene_token'],  # temporal related info
            'lidar2ego_translation': cs_record['translation'],
            'lidar2ego_rotation': cs_record['rotation'],
            'ego2global_translation': pose_record['translation'],
            'ego2global_rotation': pose_record['rotation'],
            'timestamp': sample['timestamp'],
        }

        l2e_r = info['lidar2ego_rotation']
        l2e_t = info['lidar2ego_translation']
        e2g_r = info['ego2global_rotation']
        e2g_t = info['ego2global_translation']
        l2e_r_mat = Quaternion(l2e_r).rotation_matrix
        e2g_r_mat = Quaternion(e2g_r).rotation_matrix

        if coop:
            # keep consistent with spd
            # indeed veh is the id1, and inf is other id.
            info['token_inf'] = sample['token']
            # id 5
            # cav_lidar_token = sample['data']['LIDAR_TOP_id_5']
            inf_sd_rec = nusc.get('sample_data', sample['data']['LIDAR_TOP_id_0'])
            inf_cs_record = nusc.get('calibrated_sensor',
                                inf_sd_rec['calibrated_sensor_token'])
            inf_pose_record = nusc.get('ego_pose', inf_sd_rec['ego_pose_token'])
            inf_l2e_t = np.array(inf_cs_record['translation']).reshape(3)
            inf_l2e_r = np.array(Quaternion(inf_cs_record['rotation']).rotation_matrix)
            inf_e2g_t = np.array(inf_pose_record['translation']).reshape(3)
            inf_e2g_r = np.array(Quaternion(inf_pose_record['rotation']).rotation_matrix)

            veh_l2e_t = np.array(l2e_t).reshape(3)
            veh_l2e_r = np.array(l2e_r_mat)
            veh_e2g_t = np.array(e2g_t).reshape(3)
            veh_e2g_r = np.array(e2g_r_mat)
            
            r = ((veh_l2e_r.T @ veh_e2g_r.T) @ (np.linalg.inv(inf_e2g_r).T @ np.linalg.inv(inf_l2e_r).T)).T
            t = (veh_l2e_t @ veh_e2g_r.T + veh_e2g_t) @ (np.linalg.inv(inf_e2g_r).T @ np.linalg.inv(inf_l2e_r).T)
            t -= (inf_e2g_t @ (np.linalg.inv(inf_e2g_r).T @ np.linalg.inv(inf_l2e_r).T) + \
                    inf_l2e_t @ (np.linalg.inv(inf_l2e_r).T))

            info['VehLidar2InfLidar_rotation'], info['VehLidar2InfLidar_translation'] = r,t
            # print(r)
            # print(t)
            # inf_lidar_token = sample['data']['LIDAR_TOP_id_5']
            # inf_info = obtain_sensor2top(nusc, inf_lidar_token, l2e_t, l2e_r_mat,
            #                              e2g_t, e2g_r_mat, 'lidar')
            # print(np.linalg.inv(inf_info["sensor2lidar_rotation"]))
            # print(-np.linalg.inv(inf_info["sensor2lidar_rotation"]) @ inf_info['sensor2lidar_translation'])
            
        if sample['next'] == '':
            frame_idx = 0
        else:
            frame_idx += 1

        # obtain 6 image's information per frame
        if mono:
            camera_types = [
                'CAM_FRONT_id_1',
            ]
        else:
            camera_types = [
                'CAM_FRONT_id_1',
                'CAM_FRONT_RIGHT_id_1',
                'CAM_FRONT_LEFT_id_1',
                'CAM_BACK_id_1',
                'CAM_BACK_LEFT_id_1',
                'CAM_BACK_RIGHT_id_1',
            ]

        for cam in camera_types:
            cam_token = sample['data'][cam]
            cam_path, _, cam_intrinsic = nusc.get_sample_data(cam_token)
            cam_info = obtain_sensor2top(nusc, cam_token, l2e_t, l2e_r_mat,
                                         e2g_t, e2g_r_mat, cam)
            cam_info.update(cam_intrinsic=cam_intrinsic)
            info['cams'].update({cam: cam_info})

        # obtain sweeps for a single key-frame
        sd_rec = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
        sweeps = []
        while len(sweeps) < max_sweeps:
            if not sd_rec['prev'] == '':
                sweep = obtain_sensor2top(nusc, sd_rec['prev'], l2e_t,
                                          l2e_r_mat, e2g_t, e2g_r_mat, 'lidar')
                sweeps.append(sweep)
                sd_rec = nusc.get('sample_data', sd_rec['prev'])
            else:
                break
        info['sweeps'] = sweeps

        # obtain annotation
        if not test:
            annotations = [
                nusc.get('sample_annotation', token)
                for token in sample['anns']
            ]
            locs = np.array([b.center for b in boxes]).reshape(-1, 3)
            dims = np.array([b.wlh for b in boxes]).reshape(-1, 3)
            rots = np.array([b.orientation.yaw_pitch_roll[0]
                             for b in boxes]).reshape(-1, 1)
            velocity = np.array(
                [nusc.box_velocity(token)[:2] for token in sample['anns']])
            # valid_flag = np.array(
            #     [(anno['num_lidar_pts'] + anno['num_radar_pts']) > 0
            #      for anno in annotations],
            #     dtype=bool).reshape(-1)
            ## filter valid annotation with cam front range
            valid_flag = valid_points_yx(locs)
            square_valid_flag = valid_points_yx_square(locs)
            
            valid_flag_record[sample['token']] = valid_flag.tolist()
            if coop:
                cav_valid_flag = np.array(cav_valid_flag_record[sample['token']])
                valid_flag = np.logical_or(valid_flag, cav_valid_flag)
                valid_flag = np.logical_and(valid_flag, square_valid_flag)
                # valid_flag = square_valid_flag

            instance_inds = [nusc.getind('instance', ann['instance_token'])
                             for ann in annotations]
            future_traj_all, future_traj_valid_mask_all = _get_future_traj_info(nusc, sample)
            instance_tokens = [ann['instance_token'] for ann in annotations]  # dtype('<U[length_of_str]')

            # convert velo from global to lidar
            for i in range(len(boxes)):
                velo = np.array([*velocity[i], 0.0])
                velo = velo @ np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(
                    l2e_r_mat).T
                velocity[i] = velo[:2]

            names = [b.name for b in boxes]
            for i in range(len(names)):
                if names[i] in NuScenesDataset.NameMapping:
                    names[i] = NuScenesDataset.NameMapping[names[i]]
            names = np.array(names)
            # instance_inds = [nusc.getind('instance', ann['instance_token']) for ann in annotations]
            # we need to convert rot to SECOND format.
            gt_boxes = np.concatenate([locs, dims, -rots - np.pi / 2], axis=1)
            assert len(gt_boxes) == len(
                annotations), f'{len(gt_boxes)}, {len(annotations)}'
            info['gt_boxes'] = gt_boxes
            info['gt_names'] = names
            info['gt_velocity'] = velocity.reshape(-1, 2)
            info['num_lidar_pts'] = np.array(
                [a['num_lidar_pts'] for a in annotations])
            info['num_radar_pts'] = np.array(
                [a['num_radar_pts'] for a in annotations])
            info['valid_flag'] = valid_flag
            info['gt_inds'] = np.array(instance_inds)
            info['gt_ins_tokens'] = np.array(instance_tokens)
            info['fut_traj'] = future_traj_all
            info['fut_traj_valid_mask'] = future_traj_valid_mask_all

            # add visibility_tokens
            # visibility_tokens = [int(anno['visibility_token'])
            visibility_tokens = [1
                                 for anno in annotations]
            info['visibility_tokens'] = np.array(visibility_tokens)

        if sample['scene_token'] in train_scenes:
            train_nusc_infos.append(info)
        else:
            val_nusc_infos.append(info)

    write_json(valid_flag_record, save_valid_flag_record_path)
    return train_nusc_infos, val_nusc_infos


def obtain_sensor2top(nusc,
                      sensor_token,
                      l2e_t,
                      l2e_r_mat,
                      e2g_t,
                      e2g_r_mat,
                      sensor_type='lidar'):
    """Obtain the info with RT matric from general sensor to Top LiDAR.

    Args:
        nusc (class): Dataset class in the nuScenes dataset.
        sensor_token (str): Sample data token corresponding to the
            specific sensor type.
        l2e_t (np.ndarray): Translation from lidar to ego in shape (1, 3).
        l2e_r_mat (np.ndarray): Rotation matrix from lidar to ego
            in shape (3, 3).
        e2g_t (np.ndarray): Translation from ego to global in shape (1, 3).
        e2g_r_mat (np.ndarray): Rotation matrix from ego to global
            in shape (3, 3).
        sensor_type (str): Sensor to calibrate. Default: 'lidar'.

    Returns:
        sweep (dict): Sweep information after transformation.
    """
    sd_rec = nusc.get('sample_data', sensor_token)
    cs_record = nusc.get('calibrated_sensor',
                         sd_rec['calibrated_sensor_token'])
    pose_record = nusc.get('ego_pose', sd_rec['ego_pose_token'])
    data_path = str(nusc.get_sample_data_path(sd_rec['token']))
    if os.getcwd() in data_path:  # path from lyftdataset is absolute path
        data_path = data_path.split(f'{os.getcwd()}/')[-1]  # relative path
    if nusc.dataroot in data_path:
        data_path = data_path.split(f'{nusc.dataroot}/')[-1]
    sweep = {
        'data_path': data_path,
        'type': sensor_type,
        'sample_data_token': sd_rec['token'],
        'sensor2ego_translation': cs_record['translation'],
        'sensor2ego_rotation': cs_record['rotation'],
        'ego2global_translation': pose_record['translation'],
        'ego2global_rotation': pose_record['rotation'],
        'timestamp': sd_rec['timestamp']
    }

    l2e_r_s = sweep['sensor2ego_rotation']
    l2e_t_s = sweep['sensor2ego_translation']
    e2g_r_s = sweep['ego2global_rotation']
    e2g_t_s = sweep['ego2global_translation']

    # obtain the RT from sensor to Top LiDAR
    # sweep->ego->global->ego'->lidar
    l2e_r_s_mat = Quaternion(l2e_r_s).rotation_matrix
    e2g_r_s_mat = Quaternion(e2g_r_s).rotation_matrix
    R = (l2e_r_s_mat.T @ e2g_r_s_mat.T) @ (
        np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T)
    T = (l2e_t_s @ e2g_r_s_mat.T + e2g_t_s) @ (
        np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T)
    T -= e2g_t @ (np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T
                  ) + l2e_t @ np.linalg.inv(l2e_r_mat).T
    sweep['sensor2lidar_rotation'] = R.T  # points @ R.T + T
    sweep['sensor2lidar_translation'] = T
    return sweep
