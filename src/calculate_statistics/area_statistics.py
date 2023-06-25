from typing import Union

import pandas
import geopandas
import jenkspy


def are_only_polygons(data: geopandas.GeoDataFrame):
    if not (data.geometry.geom_type.unique()[0] == 'Polygon'
            or data.geometry.geom_type.unique()[0] == 'MultiPolygon'):
        raise ValueError('Geometry types must be polygons or multipolygons')
    if data.geometry.geom_type.unique().size > 1:
        raise ValueError('Only one geometry type is allowed')


def calculate_equal_interval(data: pandas.Series, number_of_breaks: int):
    return (data.max() - data.min()) / number_of_breaks


def calculate_equal_interval_breaks(data: pandas.Series, number_of_breaks: int, remove_max_and_min: bool = False):
    interval = calculate_equal_interval(data, number_of_breaks)
    equal_interval_breaks = [data.min() + i * interval for i in range(number_of_breaks + 1)]
    if remove_max_and_min:
        equal_interval_breaks.pop(0)
        equal_interval_breaks.pop(-1)
    return equal_interval_breaks


def calculate_jenks_breaks(data: pandas.Series, number_of_breaks: int, remove_max_and_min: bool = False):
    sorted_data = data.sort_values()
    jenks = jenkspy.jenks_breaks(sorted_data, n_classes=number_of_breaks)
    if remove_max_and_min:
        jenks.pop(0)
        jenks.pop(-1)
    return jenks


class AreaStatistics:

    def __init__(self, data: geopandas.GeoDataFrame):
        are_only_polygons(data)
        self.data = data

    def get_area_statistics(self):
        areas = self.data.area

        area_statistics = {
            'sum': areas.sum(),
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

        return area_statistics

    def add_classifications_to_data(self, area_field_name: str = 'area'):
        minimum = self.get_area_statistics()['minimum']
        maximum = self.get_area_statistics()['maximum']
        self.classify_areas(
            area_field_name=area_field_name,
            breaks=[minimum] + self.get_area_statistics()['jenks'] + [maximum],
            labels=['A', 'B', 'C', 'D'],
            new_column_name='jenks'
        )
        self.classify_areas(
            area_field_name=area_field_name,
            breaks=[minimum] + self.get_area_statistics()['equal_interval_breaks'] + [maximum],
            labels=['A', 'B', 'C', 'D'],
            new_column_name='equal_interval_breaks'
        )
        self.classify_areas(
            area_field_name=area_field_name,
            breaks=[minimum,
                    self.get_area_statistics()['first_quartile'],
                    self.get_area_statistics()['second_quartile'],
                    self.get_area_statistics()['third_quartile'],
                    maximum],
            labels=['A', 'B', 'C', 'D'],
            new_column_name='quartiles'
        )

    def classify_areas(self,
                       area_field_name: str,
                       breaks: list[Union[int, float]],
                       labels: list[str],
                       new_column_name: str = 'area_class'):
        # TODO: Investigate why this warning is raised, and try to fix it
        pandas.options.mode.chained_assignment = None
        if self.data.columns.values.tolist().__contains__(area_field_name):
            series = self.data[area_field_name]
        else:
            series = self.data.area
        classification = pandas.cut(series, bins=breaks, labels=labels, include_lowest=True)
        self.data[new_column_name] = classification.astype(str)
        # TODO: Investigate why this warning is raised, and try to fix it
        pandas.options.mode.chained_assignment = 'warn'

    def calculate_area(self, column_name: str = 'area'):
        self.data[column_name] = self.data.area


class AreaStatisticsComparisonWithSampleArea(AreaStatistics):

    def __init__(self, data: geopandas.GeoDataFrame, sample_area_size: float):
        super().__init__(data)
        self.sample_area_size = sample_area_size

    def get_area_statistics(self):
        area_statistics = super().get_area_statistics()
        area_statistics['sample_area_size'] = self.sample_area_size
        area_statistics['experimental_area_ratio'] = (area_statistics['sum'] / self.sample_area_size) * 100
        return area_statistics
