import os

_project_path = '/tmp/hagDe/'


def _project(base):
    return os.path.join(_project_path, base)


config = {
    # Experiment settings
    'data_source': _project('dataset/'),
    'base_clf_dir': _project('base_clf_models/'),
    'model_save_dir': _project('checkpoints/'),
    'hagDe_result_dir': _project('results/hagDe/')
}
