import json
import sys

from calculate_statistics.basic_area_statistics import BasicAreaStatistics


def main():
    file = sys.argv[1]
    layer = sys.argv[2]
    basic_area_statistics = BasicAreaStatistics(file, layer)
    basic_statistics = basic_area_statistics.get_basic_statistics()
    result_file_name = f'basic_statistics_{layer}.json'
    with open(result_file_name, 'w') as result_file:
        result_file.write(json.dumps(basic_statistics, indent=4))


if __name__ == '__main__':
    main()
