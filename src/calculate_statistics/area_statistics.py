from typing import Union

import numpy
import pandas
import geopandas
import jenkspy
from matplotlib import pyplot
from matplotlib.lines import Line2D

from src.utils.collection_utils.list_creator import string_list_generator
from src.utils.math_utils.rounding_utils import round_up, get_scale_of_number
import src.utils.languages.languages as languages


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
) \
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
) \
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
            number_of_equal_intervals: int = 4,
            language: str = 'en'
    ):
        if is_contains_only_polygons(data):
            self.data = data
            self.number_of_natural_breaks = number_of_natural_breaks
            self.number_of_equal_intervals = number_of_equal_intervals
            self.language = languages.get_language(language)

    def get_area_statistics(self) -> dict[str, int | float | list[int | float]]:
        areas = self.data.area

        area_statistics = {
            self.language['sum']: areas.sum(),
            self.language['count']: len(areas),
            self.language['mean']: areas.mean(),
            self.language['median']: areas.median(),
            self.language['std']: areas.std(),
            self.language['var']: areas.var(),
            self.language['minimum']: areas.min(),
            self.language['first_quartile']: areas.quantile(0.25),
            self.language['second_quartile']: areas.quantile(0.5),
            self.language['third_quartile']: areas.quantile(0.75),
            self.language['jenks']: calculate_jenks_breaks(areas, self.number_of_natural_breaks, True),
            self.language['equal_interval']: calculate_equal_interval(areas, self.number_of_equal_intervals),
            self.language['equal_interval_breaks']: calculate_equal_interval_breaks(
                areas,
                self.number_of_equal_intervals,
                True
            ),
            self.language['maximum']: areas.max(),
        }

        return area_statistics

    def add_area_classifications_to_data(self, area_field_name: str = None) -> None:
        if area_field_name is None:
            area_field_name = self.language['area']
        self.classify_areas(
            area_field_name=area_field_name,
            breaks=self.get_area_statistics()[self.language['jenks']],
            labels=string_list_generator(self.language['jenks'] + ' ', self.number_of_natural_breaks),
            new_column_name=self.language['jenks']
        )
        self.classify_areas(
            area_field_name=area_field_name,
            breaks=self.get_area_statistics()[self.language['equal_interval_breaks']],
            labels=string_list_generator(self.language['equal_interval'] + ' ', self.number_of_equal_intervals),
            new_column_name=self.language['equal_interval_breaks']
        )
        self.classify_areas(
            area_field_name=area_field_name,
            breaks=[self.get_area_statistics()[self.language['first_quartile']],
                    self.get_area_statistics()[self.language['second_quartile']],
                    self.get_area_statistics()[self.language['third_quartile']]],
            labels=string_list_generator(self.language['quartile'] + ' ', 4),
            new_column_name=self.language['quartiles']
        )

    def get_classification_area_statistics(
            self,
            classification_column_name: str,
            area_field_name: str = None,
            sample_area: float = None
    ) \
            -> pandas.DataFrame:
        if area_field_name is None:
            area_field_name = self.language['area']
        data_frame = pandas.DataFrame()
        classes = pandas.Series(self.data[classification_column_name].value_counts())
        data_frame[self.language['classes']] = classes.index
        data_frame[self.language['count']] = classes.values
        summarized_areas_by_classification = pandas.Series(
            self.data.groupby(classification_column_name)[area_field_name].sum())
        data_frame[self.language['area']] = summarized_areas_by_classification.values
        class_average_area = summarized_areas_by_classification / classes
        data_frame[self.language['class_average_area']] = class_average_area.values
        area_ratio = (summarized_areas_by_classification / summarized_areas_by_classification.sum()) * 100
        data_frame[self.language['area_ratio']] = area_ratio.values
        if sample_area is not None:
            sample_area_ratio = (summarized_areas_by_classification / sample_area) * 100
            data_frame[self.language['sample_area_ratio']] = sample_area_ratio.values
        data_frame.sort_values(by=[self.language['classes']], inplace=True)
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
            diagram_title: str = None,
            x_label: str = None,
            y_label: str = None,
            y_label_2: str = None,
            x_ticks: list = None,
            sample_area_ratio_plot_color: str = 'red',
            sample_area_ration_plot_marker: str = 'o',
            sample_area_ratio_plot_line_style: str = 'solid',
            sample_area_ratio_plot_label: str = None,
            area_ratio_plot_color: str = 'blue',
            area_ratio_plot_marker: str = '^',
            area_ratio_plot_line_style: str = 'dashed',
            area_ratio_plot_label: str = None,
            legend_location: str = 'upper right',
    ) \
            -> None:
        if diagram_title is None:
            diagram_title = self.language['area_classification']
        if sample_area_ratio_plot_label is None:
            sample_area_ratio_plot_label = self.language['sample_area_ratio'] + ' (%)'
        if area_ratio_plot_label is None:
            area_ratio_plot_label = self.language['area_ratio'] + ' (%)'
        if x_label is None:
            x_label = self.language['classes']
        if y_label is None:
            y_label = self.language['count']
        if y_label_2 is None:
            y_label_2 = self.language['area_ratio'] + ' (%)'
        classification_statistics = self.get_classification_area_statistics(
            classification_column_name,
            area_field_name,
            sample_area
        )
        figure, ax = pyplot.subplots(figsize=size)
        ax.bar(
            classification_statistics[self.language['classes']],
            classification_statistics[self.language['count']],
            color=bar_color
        )
        pyplot.title(diagram_title, y=1.08)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        if x_ticks is None:
            x_ticks = numpy.arange(0, len(classification_statistics), 1)
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(classification_statistics[self.language['classes']])
        max_count = int(classification_statistics[self.language['count']].max())
        step_base = int(max_count / 10)
        y_axis_ticks = create_axis_ticks(max_count, step_base)
        ax.set_ylim(0, classification_statistics[self.language['count']].max())
        ax.set_yticks(y_axis_ticks)
        ax.set_yticklabels(y_axis_ticks)
        ax2 = ax.twinx()
        if sample_area is not None:
            ax2.plot(
                classification_statistics[self.language['classes']],
                classification_statistics[self.language['sample_area_ratio']],
                color=sample_area_ratio_plot_color,
                linestyle=sample_area_ratio_plot_line_style,
                marker=sample_area_ration_plot_marker
            )
        ax2.plot(
            classification_statistics[self.language['classes']],
            classification_statistics[self.language['area_ratio']],
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
            area_field_name: str = None,
            sample_area: float = None,
            size: tuple = (10, 5),
            path: str = None,
            dpi: int = 300,
            diagram_title: str = None,
    ) \
            -> None:
        """Partially based on https://matplotlib.org/3.1.1/gallery/pie_and_polar_charts/pie_features.html"""
        if area_field_name is None:
            area_field_name = self.language['area']
        if diagram_title is None:
            diagram_title = self.language['pie_chart_diagram_title']
        classification_statistics = self.get_classification_area_statistics(
            classification_column_name,
            area_field_name,
            sample_area
        )
        figure, ax = pyplot.subplots(figsize=size)
        patches, texts = ax.pie(
            classification_statistics[self.language['area_ratio']],
            startangle=-40,
        )
        bbox_props = dict(boxstyle='square,pad=0.3', fc='w', ec='k', lw=0.72)
        kw = dict(arrowprops=dict(arrowstyle='-'),
                  bbox=bbox_props, zorder=0, va='center')

        for i, p in enumerate(patches):
            angle = (p.theta2 - p.theta1) / 2. + p.theta1
            y = numpy.sin(numpy.deg2rad(angle))
            x = numpy.cos(numpy.deg2rad(angle))
            horizontalalignment = {-1: 'right', 1: 'left'}[int(numpy.sign(x))]
            connectionstyle = f'angle,angleA=0,angleB={angle}'
            kw['arrowprops'].update({'connectionstyle': connectionstyle})
            ax.annotate(
                f'{classification_statistics[self.language["classes"]][i]}: ' +
                f'{classification_statistics[self.language["area_ratio"]][i]:.2f} %',
                xy=(x, y),
                xytext=(1.35 * numpy.sign(x), 1.4 * y),
                horizontalalignment=horizontalalignment, **kw)
        pyplot.title(diagram_title, y=1.08)
        if path is not None:
            pyplot.savefig(path, dpi=dpi, bbox_inches='tight')
        pyplot.show()

    def classify_areas(
            self,
            breaks: list[Union[int, float]],
            labels: list[str],
            area_field_name: str = None,
            new_column_name: str = None
    ) -> None:
        if new_column_name is None:
            new_column_name = self.language['area_class']
        if area_field_name is None:
            area_field_name = self.language['area']
        if not self.data.columns.values.tolist().__contains__(area_field_name):
            self.calculate_area(area_field_name)
        bins = [self.data[area_field_name].min()] + breaks + [self.data[area_field_name].max()]
        classification = pandas.cut(x=self.data[area_field_name],
                                    bins=bins,
                                    labels=labels,
                                    ordered=False,
                                    include_lowest=True)
        classification = classification.astype(str)
        self.data.insert(loc=len(self.data.columns), column=new_column_name, value=classification)

    def calculate_area(self, column_name: str = None) -> None:
        if column_name is None:
            column_name = self.language['area']
        self.data[column_name] = self.data.area


class AreaStatisticsComparisonWithSampleArea(AreaStatistics):

    def __init__(
            self,
            data: geopandas.GeoDataFrame,
            sample_area_size: float,
            number_of_natural_breaks: int = 4,
            number_of_equal_intervals: int = 4,
            language: str = 'en'
    ):
        super().__init__(data, number_of_natural_breaks, number_of_equal_intervals, language)
        self.sample_area_size = sample_area_size

    def get_area_statistics(self):
        area_statistics = super().get_area_statistics()
        area_statistics[self.language['sample_area_size']] = self.sample_area_size
        area_statistics[self.language['experimental_area_ratio']] = \
            (area_statistics[self.language['sum']] / self.sample_area_size) * 100
        return area_statistics
