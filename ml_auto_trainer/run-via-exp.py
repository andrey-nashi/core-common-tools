import json
from core.engine.experiment import Experiment, ExperimentConfiguration

import albumentations as alb

def run(path_experiment_json: str):

    experiment_cfg = ExperimentConfiguration()
    experiment_cfg.load_from_file(path_experiment_json)

    experiment = Experiment()
    experiment.build_train(experiment_cfg)



if __name__ == '__main__':
    path = "./experiment-cfg.json"
    run(path)