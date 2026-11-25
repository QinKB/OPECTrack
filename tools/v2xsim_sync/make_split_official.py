"""Convert the official V2X-Sim scene index split to scene names."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('split_dir', type=Path)
    parser.add_argument('scene_table', type=Path)
    parser.add_argument('out', type=Path)
    args = parser.parse_args()
    names = [scene['name'] for scene in json.loads(args.scene_table.read_text())]
    if names != [f'scene_{i}' for i in range(1, 101)]:
        raise ValueError('Expected the 100 V2X-Sim 2.0 scenes in release order')
    indices = {
        split: sorted(int(line) for line in (args.split_dir / f'{split}_scenes.txt').read_text().splitlines()
                      if line.strip())
        for split in ('train', 'val', 'test')
    }
    if ([len(indices[split]) for split in ('train', 'val', 'test')] != [80, 10, 10]
            or sorted(i for split in indices.values() for i in split) != list(range(len(names)))):
        raise ValueError('Expected the official disjoint 80/10/10 split covering all 100 scenes')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    batch_split = {split: [names[i] for i in members] for split, members in indices.items()}
    args.out.write_text(json.dumps({'batch_split': batch_split}, indent=1))


if __name__ == '__main__':
    main()
