
from core.engine.experiment import Experiment, ExperimentConfiguration
from core.engine.engine import Engine


def run(path_experiment_json: str):

    experiment_cfg = ExperimentConfiguration()
    experiment_cfg.load_from_file(path_experiment_json)

    experiment = Experiment()
    experiment.build_train(experiment_cfg)

    Engine.run_trainer(experiment)


if __name__ == '__main__':
    path = "./experiment-cfg.json"
    run(path)