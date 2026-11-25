"""Build the V2X-Sim tracking and roadside datasets for agents 1 and 0."""
import json
import shutil
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from pyquaternion import Quaternion

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'projects' / 'mmdet3d_plugin'))
from opectrack.data import spd_pairs  # noqa: E402

INFOS, SHARDS, TABLES, RAW, SPLIT, OUT = sys.argv[1:7]
OUT = Path(OUT)
T = 1
CLS = {'vehicle.car': 'car', 'vehicle.emergency.police': 'car', 'vehicle.bicycle': 'bicycle',
       'vehicle.motorcycle': 'bicycle'}
GROUP = {'car': 0, 'bicycle': 2}
split = json.load(open(SPLIT))['batch_split']

# Per-annotation point counts and geometry from all shards.
F = {}
for k in range(20):
    F.update(pickle.load(open(f'{SHARDS}/s{k}/data/gt_fields.pkl', 'rb')))


def masks(f):
    cls = np.array([CLS.get(c, '') for c in f['category']])
    ok = (cls != '') & ~f['size_defect']
    nv, ni = np.maximum(f['npts_veh'], 0), np.maximum(f['npts_inf'], 0)
    ego = ok & ~f['is_ego'] & (f['d_ego'] < 50.0) & ((nv >= T) | (ni >= T))
    rsu = ok & (np.abs(f['rsu_x']) <= 51.2) & (np.abs(f['rsu_y']) <= 51.2) & (ni >= T)
    return cls, ego, rsu, nv, ni


def restrict(info, m, cls, npts):
    out = dict(info)
    for k in ('gt_boxes', 'gt_velocity', 'gt_inds', 'gt_ins_tokens', 'fut_traj', 'fut_traj_valid_mask',
              'visibility_tokens', 'num_radar_pts'):
        if k in info:
            out[k] = info[k][m]
    out['gt_names'] = cls[m].astype('<U8')
    out['num_lidar_pts'] = npts[m].astype(np.int64)
    out['valid_flag'] = np.ones(int(m.sum()), bool)
    out['timestamp_inf'] = info['timestamp']
    out['token_inf'] = info['token']
    return out


def lidar2world(info):
    m1 = np.eye(4); m1[:3, :3] = Quaternion(info['lidar2ego_rotation']).rotation_matrix
    m1[:3, 3] = info['lidar2ego_translation']
    m2 = np.eye(4); m2[:3, :3] = Quaternion(info['ego2global_rotation']).rotation_matrix
    m2[:3, 3] = info['ego2global_translation']
    return m2 @ m1


keep_ego, keep_rsu = {}, {}          # sample token -> ordered list of kept annotation positions
order_tokens = {}                     # sample token -> instance tokens in info order (checked against the tables)
for part in ('train', 'val', 'test'):
    co = pickle.load(open(f'{INFOS}/V2X-Sim-full-coop/nuscenes_infos_temporal_{part}.pkl', 'rb'))
    i0 = pickle.load(open(f'{INFOS}/V2X-Sim-full-id0/nuscenes_infos_temporal_{part}.pkl', 'rb'))
    b0 = {i['token']: i for i in i0['infos']}
    out_co, out_i0, frames = [], [], defaultdict(list)
    for info in co['infos']:
        tok = info['token']; f = F[tok]; j = b0[tok]
        assert np.array_equal(info['gt_ins_tokens'], j['gt_ins_tokens'])
        assert len(f['category']) == len(info['gt_boxes'])
        cls, ego, rsu, nv, ni = masks(f)
        keep_ego[tok] = np.nonzero(ego)[0]; keep_rsu[tok] = np.nonzero(rsu)[0]
        order_tokens[tok] = info['gt_ins_tokens']
        out_co.append(restrict(info, ego, cls, nv + ni))
        out_i0.append(restrict(j, rsu, cls, ni))
        # frame pair (opectrack.data.spd_pairs.build structure)
        v2w, i2w = lidar2world(info), lidar2world(j)
        vis_v = ego & (nv >= T)

        def boxes(src, m):
            g = src['gt_boxes'][m]
            return dict(xyz=g[:, :3].copy(), size=g[:, [4, 3, 5]].copy(), yaw=-g[:, 6] - np.pi / 2,
                        label=np.array([GROUP[c] for c in cls[m]], dtype=np.int64),
                        track=src['gt_inds'][m].astype(np.int64), token=list(src['gt_ins_tokens'][m]))
        v, i_ = boxes(info, vis_v), boxes(j, rsu)
        vi = {t: k for k, t in enumerate(v['token'])}
        pairs = [(vi[t], k) for k, t in enumerate(i_['token']) if t in vi]
        scene = 'scene_' + info['lidar_path'].split('/')[-1].split('_')[1]
        frames[scene].append(dict(
            sequence=scene, vehicle_frame=tok, infrastructure_frame=tok, time_v=info['timestamp'] * 1e-6,
            time_i=info['timestamp'] * 1e-6, v2w=v2w, i2w=i2w, i2v=np.linalg.inv(v2w) @ i2w, v=v, i=i_,
            pairs=np.array(pairs, dtype=np.int64).reshape(-1, 2), dataset_offset=np.zeros(2), intersection=''))
    for fr in frames.values():
        fr.sort(key=lambda e: e['time_v'])
        spd_pairs._velocities(fr, 'v', 'v2w')
        spd_pairs._velocities(fr, 'i', 'i2w')
        for k, x in enumerate(fr):
            prev = fr[max(k - 1, 0)]
            x['relative'] = np.linalg.inv(x['v2w']) @ prev['v2w']
            dt = x['time_v'] - prev['time_v']
            step = (x['v2w'][:3, 3] - prev['v2w'][:3, 3]) / dt if dt > 1e-3 else np.zeros(3)
            x['ego_velocity'] = (x['v2w'][:3, :3].T @ step)[:2]
    for view, infos in (('cooperative', out_co), ('infrastructure-side', out_i0)):
        d = OUT / 'data/infos' / view; d.mkdir(parents=True, exist_ok=True)
        with open(d / f'spd_infos_temporal_{part}.pkl', 'wb') as fh:
            pickle.dump(dict(infos=infos, metadata=co['metadata']), fh, protocol=4)
    (OUT / 'pairs').mkdir(parents=True, exist_ok=True)
    torch.save(dict(frames), OUT / 'pairs' / f'pairs_{part}.pt')
    del co, i0, b0, out_co, out_i0, frames

# tables: annotations of the ground truth only, in info order
J = lambda n: json.load(open(f'{TABLES}/{n}'))  # noqa: E731
anns = J('sample_annotation.json')
samples = {s['token']: s for s in J('sample.json')}
by_sample = defaultdict(list)
for a in anns:
    by_sample[a['sample_token']].append(a)
del anns
cats = J('category.json')
insts = J('instance.json')
for tok, lst in by_sample.items():          # the infos were written in this order: check
    assert [a['instance_token'] for a in lst] == list(order_tokens[tok]), tok
for view, keep in (('cooperative', keep_ego), ('infrastructure-side', keep_rsu)):
    d = OUT / 'datasets' / view / 'v1.0-trainval'; d.mkdir(parents=True, exist_ok=True)
    kept = []
    for tok, lst in by_sample.items():
        f = F[tok]; nv, ni = np.maximum(f['npts_veh'], 0), np.maximum(f['npts_inf'], 0)
        for k in keep[tok]:
            a = dict(lst[k]); a['num_lidar_pts'] = int((nv[k] + ni[k]) if view == 'cooperative' else ni[k])
            a['num_radar_pts'] = 0
            kept.append(a)
    per_inst = defaultdict(list)
    for a in kept:
        per_inst[a['instance_token']].append(a)
    for lst in per_inst.values():
        lst.sort(key=lambda a: samples[a['sample_token']]['timestamp'])
        for k, a in enumerate(lst):
            a['prev'] = lst[k - 1]['token'] if k else ''
            a['next'] = lst[k + 1]['token'] if k + 1 < len(lst) else ''
    new_insts = []
    for i in insts:
        lst = per_inst.get(i['token'])
        if lst:
            new_insts.append(dict(i, nbr_annotations=len(lst), first_annotation_token=lst[0]['token'],
                                  last_annotation_token=lst[-1]['token']))
    json.dump(kept, open(d / 'sample_annotation.json', 'w'))
    json.dump(new_insts, open(d / 'instance.json', 'w'))
    json.dump([dict(c, name=CLS.get(c['name'], c['name'])) for c in cats], open(d / 'category.json', 'w'))
    for n in ('scene.json', 'sample.json', 'sample_data.json', 'ego_pose.json', 'calibrated_sensor.json',
              'attribute.json', 'visibility.json', 'log.json'):
        shutil.copyfile(Path(TABLES) / n, d / n)
    sensor = open(f'{RAW}/sensor.json').read()
    sensor = sensor.replace('LIDAR_TOP_id_1' if view == 'cooperative' else 'LIDAR_TOP_id_0', 'LIDAR_TOP')
    open(d / 'sensor.json', 'w').write(sensor)
sd = OUT / 'data/split_datas'; sd.mkdir(parents=True, exist_ok=True)
for ev in ('val', 'test'):
    json.dump({'batch_split': {'train': split['train'], 'val': split[ev], 'test': split[ev]}},
              open(sd / f'v2x-sim-split-{ev}.json', 'w'), indent=1)
