"""Copy scene tables in release order and name the selected LiDAR channel LIDAR_TOP."""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from jsonstream import iter_json_array  # noqa: E402

SMALL = ['sensor.json', 'calibrated_sensor.json', 'category.json', 'attribute.json', 'visibility.json', 'log.json']


class ArrayWriter:
    def __init__(self, path):
        self.f = open(path, 'w')
        self.f.write('[\n')
        self.n = 0

    def write(self, obj):
        if self.n:
            self.f.write(',\n')
        json.dump(obj, self.f)
        self.n += 1

    def close(self):
        self.f.write('\n]\n')
        self.f.close()
        return self.n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--raw', required=True, help='raw V2X-Sim 2.0 table directory (the extracted v2.0/)')
    ap.add_argument('--out', required=True, help='output <dataset root>/v1.0-trainval directory')
    ap.add_argument('--ego-id', type=int, required=True, help='agent whose LIDAR_TOP_id_<k> becomes LIDAR_TOP')
    ap.add_argument('--scenes', nargs='*', default=None, help='scene names to keep (default: all)')
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=False)
    R = lambda n: os.path.join(args.raw, n)  # noqa: E731
    W = lambda n: os.path.join(args.out, n)  # noqa: E731
    counts = {}

    for n in SMALL:
        txt = open(R(n)).read()
        if n == 'sensor.json':
            src = f'LIDAR_TOP_id_{args.ego_id}'
            assert f'"{src}"' in txt, src
            txt = txt.replace(src, 'LIDAR_TOP')
        open(W(n), 'w').write(txt)

    scenes = json.load(open(R('scene.json')))
    if args.scenes is not None:
        known = {s['name'] for s in scenes}
        missing = [s for s in args.scenes if s not in known]
        if missing:
            raise SystemExit(f'unknown scenes: {missing}')
        scenes = [s for s in scenes if s['name'] in set(args.scenes)]
    scene_tok = {s['token'] for s in scenes}
    json.dump(scenes, open(W('scene.json'), 'w'))
    counts['scene'] = len(scenes)

    samples = [s for s in json.load(open(R('sample.json'))) if s['scene_token'] in scene_tok]
    sample_tok = {s['token'] for s in samples}
    json.dump(samples, open(W('sample.json'), 'w'))
    counts['sample'] = len(samples)

    w = ArrayWriter(W('sample_data.json'))
    ep_tok = set()
    for o in iter_json_array(R('sample_data.json')):
        if o['sample_token'] in sample_tok:
            w.write(o)
            ep_tok.add(o['ego_pose_token'])
    counts['sample_data'] = w.close()

    w = ArrayWriter(W('ego_pose.json'))
    for o in iter_json_array(R('ego_pose.json')):
        if o['token'] in ep_tok:
            w.write(o)
    counts['ego_pose'] = w.close()

    w = ArrayWriter(W('sample_annotation.json'))
    inst_tok = set()
    for o in iter_json_array(R('sample_annotation.json')):
        if o['sample_token'] in sample_tok:
            w.write(o)
            inst_tok.add(o['instance_token'])
    counts['sample_annotation'] = w.close()

    insts = [i for i in json.load(open(R('instance.json'))) if i['token'] in inst_tok]
    json.dump(insts, open(W('instance.json'), 'w'))
    counts['instance'] = len(insts)

    print(json.dumps(counts))


if __name__ == '__main__':
    main()
