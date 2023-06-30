import pandas
from pathlib import Path


def validate_excel_file_name(file_name: str):
    if Path(file_name).suffix != '.xlsx':
        raise ValueError(f'File name must end with .xlsx, your file name is {file_name}')


def set_mode_based_on_file_existence(file_name: str) -> str:
    if Path(file_name).exists():
        return 'a'
    return 'w'


def update_excel_sheet_name_if_exists(file_name: str, sheet_name: str) -> str:
    validate_excel_file_name(file_name)
    if Path(file_name).exists():
        file = pandas.ExcelFile(file_name)
        sheets = file.sheet_names
        while sheet_name in sheets:
            sheet_name = sheet_name + '_01'
    return sheet_name


def write_excel_sheet_from_dict(dictionary: dict, file_name: str, sheet_name: str = 'Sheet1'):
    validate_excel_file_name(file_name)
    mode = set_mode_based_on_file_existence(file_name)
    sheet_name = update_excel_sheet_name_if_exists(file_name, sheet_name)
    with pandas.ExcelWriter(file_name, mode=mode) as excel_writer:
        pandas.DataFrame.from_dict(dictionary, orient='index').to_excel(excel_writer, sheet_name=sheet_name)


def write_excel_sheet_from_dataframe(dataframe: pandas.DataFrame, file_name: str, sheet_name: str = 'Sheet1'):
    validate_excel_file_name(file_name)
    mode = set_mode_based_on_file_existence(file_name)
    sheet_name = update_excel_sheet_name_if_exists(file_name, sheet_name)
    with pandas.ExcelWriter(file_name, mode=mode) as excel_writer:
        dataframe.to_excel(excel_writer, sheet_name=sheet_name)
