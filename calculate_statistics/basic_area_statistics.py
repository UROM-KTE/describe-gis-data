import pandas
import geopandas
import jenkspy


def calculate_equal_interval(data: pandas.Series, number_of_breaks: int):
    return (data.max() - data.min()) / number_of_breaks


def calculate_jenks_breaks(data: pandas.Series, number_of_breaks: int, remove_max_and_min: bool = False):
    sorted_data = data.sort_values()
    jenks = jenkspy.jenks_breaks(sorted_data, n_classes=number_of_breaks)
    if remove_max_and_min:
        jenks.pop(0)
        jenks.pop(-1)
    return jenks


def calculate_equal_interval_breaks(data: pandas.Series, number_of_breaks: int, remove_max_and_min: bool = False):
    interval = calculate_equal_interval(data, number_of_breaks)
    equal_interval_breaks = [data.min() + i * interval for i in range(number_of_breaks + 1)]
    if remove_max_and_min:
        equal_interval_breaks.pop(0)
        equal_interval_breaks.pop(-1)
    return equal_interval_breaks


class BasicAreaStatistics:

    def __init__(self, file: str, layer: str):
        self.file = file
        self.layer = layer

    def get_basic_statistics(self):
        print(self.file, self.layer)
        data = geopandas.read_file(self.file, layer=self.layer)
        areas = data.area

        basic_statistics = {
            'count': len(areas),
            'mean': areas.mean(),
            'median': areas.median(),
            'std': areas.std(),
            'minimum': areas.min(),
            'first_quartile': areas.quantile(0.25),
            'second_quartile': areas.quantile(0.5),
            'third_quartile': areas.quantile(0.75),
            'jenks': calculate_jenks_breaks(areas, 4, True),
            'equal_interval': calculate_equal_interval(areas, 4),
            'equal_interval_breaks': calculate_equal_interval_breaks(areas, 4, True),
            'maximum': areas.max(),
        }

        return basic_statistics
