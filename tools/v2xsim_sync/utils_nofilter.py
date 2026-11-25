"""Prepare the selected V2X-Sim scene tables with LET-VIC's table transforms."""
import argparse
from pathlib import Path

from tools.v2xsim_data_converter import v2xsim_utils


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    args = parser.parse_args()
    tables = args.root / 'v1.0-trainval'
    read = v2xsim_utils.load_json
    instances, annotations = v2xsim_utils.process_instance(
        read(tables / 'instance.json'), read(tables / 'sample_annotation.json'))
    annotations = v2xsim_utils.process_sample_annotations(annotations)
    samples, sample_data, poses = v2xsim_utils.process_timestamp(
        read(tables / 'sample.json'), read(tables / 'sample_data.json'),
        read(tables / 'ego_pose.json'))
    logs = v2xsim_utils.process_log(read(tables / 'log.json'))
    for name, rows in {'instance': instances, 'sample_annotation': annotations,
                       'sample': samples, 'sample_data': sample_data, 'ego_pose': poses,
                       'log': logs}.items():
        v2xsim_utils.write_json(rows, tables / f'{name}.json')


if __name__ == '__main__':
    main()
