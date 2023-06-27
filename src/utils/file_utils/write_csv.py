import csv


def write_csv_from_dict(dictionary: dict, file_name: str):
    with open(file_name, 'w') as csv_file:
        csv_writer = csv.writer(csv_file)
        for key, value in dictionary.items():
            csv_writer.writerow([key, value])
