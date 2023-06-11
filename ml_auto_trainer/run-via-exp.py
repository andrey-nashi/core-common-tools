
from core.engine.experiment import Experiment, ExperimentConfiguration
from core.engine.engine import Engine


def run(path_experiment_json: str):

    # ---- Load experiment configurations
    experiment_cfg = ExperimentConfiguration()
    experiment_cfg.load_from_file(path_experiment_json)

    # ---- Experiment: train
    experiment = Experiment()
    experiment.build_train(experiment_cfg)
    Engine.run_trainer(experiment)

    # ---- Experiment: test
    experiment = Experiment()
    experiment.build_test(experiment_cfg)
    Engine.run_tester(experiment)

if __name__ == '__main__':
    path = "examples/experiment-cfg.json"
    run(path)