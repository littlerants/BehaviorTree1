import argparse
import os
import sys
import time

import carla


def main():
    argparser = argparse.ArgumentParser(
        description=__doc__)
    argparser.add_argument(
        '-x', '--xodr-path',
        metavar='XODR_FILE_PATH',
        help='load a new map with a minimum physical road representation of the provided OpenDRIVE')
    argparser.add_argument(
        '-m', '--map-name',
        metavar='XODR_MAP_NAME',
        help='specify map name')
    args = argparser.parse_args()

    client = carla.Client("localhost", 2000)
    client.set_timeout(180)

    if os.path.exists(args.xodr_path):
        with open(args.xodr_path, encoding='utf-8') as od_file:
            try:
                data = od_file.read()
            except OSError:
                sys.exit()

        client.generate_opendrive_world(
            data, carla.OpendriveGenerationParameters(
                map_name=args.map_name,
                map_type="highway"))

        while True:
            status = client.get_generation_status()
            print(status)
            if status >= 100.00:
                break
            time.sleep(0.1)


if __name__ == '__main__':
    main()
