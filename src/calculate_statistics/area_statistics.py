from typing import Union

import numpy
import pandas
import geopandas
import jenkspy
from matplotlib import pyplot
from matplotlib.lines import Line2D

from src.utils.collection_utils.list_creator import string_list_generator
from src.utils.math_utils.rounding_utils import round_up, get_scale_of_number


def is_contains_only_polygons(data: geopandas.GeoDataFrame) -> bool:
    if not (data.geometry.geom_type.unique()[0] == 'Polygon'
            or data.geometry.geom_type.unique()[0] == 'MultiPolygon'):
        raise ValueError('Geometry types must be polygons or multipolygons')
    if data.geometry.geom_type.unique().size > 1:
        raise ValueError('Only one geometry type is allowed')
    return True


def calculate_equal_interval(data: pandas.Series, number_of_breaks: int) -> Union[int, float]:
    return (data.max() - data.min()) / number_of_breaks


def calculate_equal_interval_breaks(
        data: pandas.Series,
        number_of_breaks: int,
        remove_max_and_min: bool = False
)\
        -> list[Union[int, float]]:
    interval = calculate_equal_interval(data, number_of_breaks)
    equal_interval_breaks = [data.min() + i * interval for i in range(number_of_breaks + 1)]
    if remove_max_and_min:
        equal_interval_breaks.pop(0)
        equal_interval_breaks.pop(-1)
    return equal_interval_breaks


def calculate_jenks_breaks(
        data: pandas.Series,
        number_of_breaks: int,
        remove_max_and_min: bool = False
)\
        -> list[Union[int, float]]:
    sorted_data = data.sort_values()
    jenks = jenkspy.jenks_breaks(sorted_data, n_classes=number_of_breaks)
    if remove_max_and_min:
        jenks.pop(0)
        jenks.pop(-1)
    return jenks


def create_axis_ticks(max_count: int, step_base: int, start: int = 0) -> numpy.ndarray:
    y_axis_ticks = numpy.arange(
        start,
        stop=max_count + round_up(step_base, get_scale_of_number(step_base) * -1) + 1,
        step=int(round_up(step_base, get_scale_of_number(step_base) * -1)))
    return y_axis_ticks


class AreaStatistics:

    def __init__(
            self,
            data: geopandas.GeoDataFrame,
            number_of_natural_breaks: int = 4,
            number_of_equal_intervals: int = 4
    ):
        if is_contains_only_polygons(data):
            self.data = data
            self.number_of_natural_breaks = number_of_natural_breaks
            self.number_of_equal_intervals = number_of_equal_intervals

    def get_area_statistics(self) -> dict[str, int | float | list[int | float]]:
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
            'jenks': calculate_jenks_breaks(areas, self.number_of_natural_breaks, True),
            'equal_interval': calculate_equal_interval(areas, self.number_of_equal_intervals),
            'equal_interval_breaks': calculate_equal_interval_breaks(areas, self.number_of_equal_intervals, True),
            'maximum': areas.max(),
        }

        return area_statistics

    def add_area_classifications_to_data(self, area_field_name: str = 'area') -> None:
        minimum = self.get_area_statistics()['minimum']
        maximum = self.get_area_statistics()['maximum']
        self.classify_areas(
            area_field_name=area_field_name,
            breaks=[minimum] + self.get_area_statistics()['jenks'] + [maximum],
            labels=string_list_generator('jenks_', self.number_of_natural_breaks),
            new_column_name='jenks'
        )
        self.classify_areas(
            area_field_name=area_field_name,
            breaks=[minimum] + self.get_area_statistics()['equal_interval_breaks'] + [maximum],
            labels=string_list_generator('equal_interval_', self.number_of_equal_intervals),
            new_column_name='equal_interval_breaks'
        )
        self.classify_areas(
            area_field_name=area_field_name,
            breaks=[minimum,
                    self.get_area_statistics()['first_quartile'],
                    self.get_area_statistics()['second_quartile'],
                    self.get_area_statistics()['third_quartile'],
                    maximum],
            labels=string_list_generator('quartile_', 4),
            new_column_name='quartiles'
        )

    def get_classification_area_statistics(
            self,
            classification_column_name: str,
            area_field_name: str = 'area',
            sample_area: float = None
    )\
            -> pandas.DataFrame:
        data_frame = pandas.DataFrame()
        classes = pandas.Series(self.data[classification_column_name].value_counts())
        data_frame['classes'] = classes.index
        data_frame['count'] = classes.values
        summarized_areas_by_classification = pandas.Series(
            self.data.groupby(classification_column_name)[area_field_name].sum())
        data_frame['area'] = summarized_areas_by_classification.values
        class_average_area = summarized_areas_by_classification / classes
        data_frame['class_average_area'] = class_average_area.values
        area_ratio = (summarized_areas_by_classification / summarized_areas_by_classification.sum()) * 100
        data_frame['area_ratio'] = area_ratio.values
        if sample_area is not None:
            sample_area_ratio = (summarized_areas_by_classification / sample_area) * 100
            data_frame['sample_area_ratio'] = sample_area_ratio.values
        data_frame.sort_values(by=['classes'], inplace=True)
        return data_frame

    def create_classification_diagram(
            self,
            classification_column_name: str,
            path: str = None,
            dpi: int = 300,
            area_field_name: str = 'area',
            sample_area: float = None,
            size: tuple = (10, 5),
            bar_color: str = '#b6d97e',
            diagram_title: str = 'Area classification',
            x_label: str = 'Classes',
            y_label: str = 'Count',
            y_label_2: str = 'Area ratio (%)',
            x_ticks: list = None,
            sample_area_ratio_plot_color: str = 'red',
            sample_area_ration_plot_marker: str = 'o',
            sample_area_ratio_plot_line_style: str = 'solid',
            sample_area_ratio_plot_label: str = 'Sample area ratio (%)',
            area_ratio_plot_color: str = 'blue',
            area_ratio_plot_marker: str = '^',
            area_ratio_plot_line_style: str = 'dashed',
            area_ratio_plot_label: str = 'Area ratio (%)',
            legend_location: str = 'upper right',
    )\
            -> None:
        classification_statistics = self.get_classification_area_statistics(
            classification_column_name,
            area_field_name,
            sample_area
        )
        figure, ax = pyplot.subplots(figsize=size)
        ax.bar(classification_statistics['classes'], classification_statistics['count'], color=bar_color)
        ax.set_title(diagram_title)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        if x_ticks is None:
            x_ticks = numpy.arange(0, len(classification_statistics), 1)
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(classification_statistics['classes'])
        max_count = int(classification_statistics['count'].max())
        step_base = int(max_count / 10)
        y_axis_ticks = create_axis_ticks(max_count, step_base)
        ax.set_ylim(0, classification_statistics['count'].max())
        ax.set_yticks(y_axis_ticks)
        ax.set_yticklabels(y_axis_ticks)
        ax2 = ax.twinx()
        if sample_area is not None:
            ax2.plot(
                classification_statistics['classes'],
                classification_statistics['sample_area_ratio'],
                color=sample_area_ratio_plot_color,
                linestyle=sample_area_ratio_plot_line_style,
                marker=sample_area_ration_plot_marker
            )
        ax2.plot(
            classification_statistics['classes'],
            classification_statistics['area_ratio'],
            color=area_ratio_plot_color,
            linestyle=area_ratio_plot_line_style,
            marker=area_ratio_plot_marker
        )
        ax2.set_ylabel(y_label_2)
        ax2.set_ylim(0, 100)
        ax2.set_yticks(numpy.arange(0, 101, 10))
        ax2.set_yticklabels(numpy.arange(0, 101, 10))
        legend_elements = [
            Line2D([0], [0], color=sample_area_ratio_plot_color, linestyle=sample_area_ratio_plot_line_style,
                   marker=sample_area_ration_plot_marker, label=sample_area_ratio_plot_label),
            Line2D([0], [0], color=area_ratio_plot_color, linestyle=area_ratio_plot_line_style,
                   marker=area_ratio_plot_marker, label=area_ratio_plot_label)
        ]
        ax2.legend(handles=legend_elements, loc=legend_location)
        if path is not None:
            pyplot.savefig(path, dpi=dpi, bbox_inches='tight')
        pyplot.show()

    def create_classification_area_ratio_pie_chart(
            self,
            classification_column_name: str,
            area_field_name: str = 'area',
            sample_area: float = None,
            path: str = None,
            dpi: int = 300,
            diagram_title: str = 'The proportion of areas covered with oleasters according to each group',
    )\
            -> None:
        classification_statistics = self.get_classification_area_statistics(
            classification_column_name,
            area_field_name,
            sample_area
        )
        pyplot.pie(
            classification_statistics['area_ratio'],
            labels=classification_statistics['classes'],
            autopct='%1.1f%%',
            startangle=90,
        )
        pyplot.title(diagram_title)
        if path is not None:
            pyplot.savefig(path, dpi=dpi, bbox_inches='tight')
        pyplot.show()

    def classify_areas(
            self,
            area_field_name: str,
            breaks: list[Union[int, float]],
            labels: list[str],
            new_column_name: str = 'area_class'
    ) -> None:
        if not self.data.columns.values.tolist().__contains__(area_field_name):
            self.data[area_field_name] = self.data.area
        classification = pandas.cut(x=self.data[area_field_name],
                                    bins=breaks,
                                    labels=labels,
                                    ordered=False,
                                    include_lowest=True)
        classification = classification.astype(str)
        self.data.insert(loc=len(self.data.columns), column=new_column_name, value=classification)

    def calculate_area(self, column_name: str = 'area') -> None:
        self.data[column_name] = self.data.area


class AreaStatisticsComparisonWithSampleArea(AreaStatistics):

    def __init__(
            self,
            data: geopandas.GeoDataFrame,
            sample_area_size: float,
            number_of_natural_breaks: int = 4,
            number_of_equal_intervals: int = 4,
    ):
        super().__init__(data, number_of_natural_breaks, number_of_equal_intervals)
        self.sample_area_size = sample_area_size

    def get_area_statistics(self):
        area_statistics = super().get_area_statistics()
        area_statistics['sample_area_size'] = self.sample_area_size
        area_statistics['experimental_area_ratio'] = (area_statistics['sum'] / self.sample_area_size) * 100
        return area_statistics
