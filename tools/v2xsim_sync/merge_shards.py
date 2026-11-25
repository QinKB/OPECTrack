"""Merge scene shards into information files for the official splits."""
import argparse
import json
import pickle
from pathlib import Path

import numpy as np

VIEWS = ('V2X-Sim-full-coop', 'V2X-Sim-full-id0')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out', type=Path)
    parser.add_argument('split_file', type=Path)
    parser.add_argument('instances', type=Path)
    parser.add_argument('shards', nargs='+', type=Path)
    args = parser.parse_args()
    splits = json.loads(args.split_file.read_text())['batch_split']
    scene_split = {scene: split for split, scenes in splits.items() for scene in scenes}
    present = {row['token'] for shard in args.shards for row in json.loads(
        (shard / 'datasets/V2X-Sim-full-coop/v1.0-trainval/instance.json').read_text())}
    instance_order = [row['token'] for row in json.loads(args.instances.read_text()) if row['token'] in present]
    reference_index = {token: i for i, token in enumerate(instance_order)}
    assert len(reference_index) == len(present)

    for view in VIEWS:
        merged = {split: [] for split in splits}
        for shard in args.shards:
            shard_instances = [row['token'] for row in json.loads(
                (shard / 'datasets' / view / 'v1.0-trainval/instance.json').read_text())]
            for part, allowed in [('train', {'train'}), ('val', {'val', 'test'})]:
                with (shard / 'data/infos' / view / f'nuscenes_infos_temporal_{part}.pkl').open('rb') as stream:
                    data = pickle.load(stream)
                metadata = data['metadata']
                for info in data['infos']:
                    scene = 'scene_' + Path(info['lidar_path']).name.split('_')[1]
                    split = scene_split[scene]
                    assert split in allowed
                    indices, tokens = info['gt_inds'], info['gt_ins_tokens']
                    assert len(indices) == len(tokens)
                    assert all(shard_instances[i] == token for i, token in zip(indices, tokens))
                    info['gt_inds'] = np.array([reference_index[token] for token in tokens], dtype=indices.dtype)
                    merged[split].append(info)
        target = args.out / view
        target.mkdir(parents=True, exist_ok=True)
        for split, infos in merged.items():
            assert len(infos) == 100 * len(splits[split]), (view, split, len(infos))
            with (target / f'nuscenes_infos_temporal_{split}.pkl').open('wb') as stream:
                pickle.dump(dict(infos=infos, metadata=metadata), stream, protocol=4)


if __name__ == '__main__':
    main()
