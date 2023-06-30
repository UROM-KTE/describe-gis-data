import os


def create_results_folder(
        results_folder: str,
        project_folder: str,
        statistics_folder: str = 'statistics',
        figures_folder: str = 'figures',
        gis_data_folder: str = 'gis_data'
) -> None:
    """ Checks and creates the results folder and its sub-folders if they do not exist """
    if not os.path.exists(results_folder):
        os.mkdir(results_folder)
    if not os.path.exists(os.path.join(results_folder, project_folder)):
        os.mkdir(os.path.join(results_folder, project_folder))
    if not os.path.exists(os.path.join(results_folder, project_folder, statistics_folder)):
        os.mkdir(os.path.join(results_folder, project_folder, statistics_folder))
    if not os.path.exists(os.path.join(results_folder, project_folder, figures_folder)):
        os.mkdir(os.path.join(results_folder, project_folder, figures_folder))
    if not os.path.exists(os.path.join(results_folder, project_folder, gis_data_folder)):
        os.mkdir(os.path.join(results_folder, project_folder, gis_data_folder))


def remove_previous_results(
        project_folder: str,
) -> None:
    """ Removes the previous results folder and its sub-folders if they exist """
    if os.path.exists(project_folder):
        os.system('rm -r ' + project_folder)
