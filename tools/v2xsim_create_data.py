"""Convert V2X-Sim tables to the LiDAR information files used by the tracker."""
import argparse
from pathlib import Path

from v2xsim_data_converter.v2xsim_nuscenes_converter import create_nuscenes_infos


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root-path', required=True)
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--coop', action='store_true')
    args = parser.parse_args()
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    create_nuscenes_infos(args.root_path, args.out_dir, args.root_path, 'nuscenes',
                          use_can_bus=False, mono=True, coop=args.coop)


if __name__ == '__main__':
    main()
