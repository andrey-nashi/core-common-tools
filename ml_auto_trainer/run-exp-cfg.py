import argparse

from core.engine import Experiment, ExperimentConfiguration
from core.engine.engine import Engine

# --------------------------------------------------------------
def run(path_experiment_json: str):

    # ---- Load experiment configurations
    experiment_cfg = ExperimentConfiguration()
    is_ok = experiment_cfg.load_from_file(path_experiment_json)
    if not is_ok:
        print("[ERROR]: Incorrect file format")
        return

    experiment = Experiment()
    experiment.configure(experiment_cfg)

    # ---- Experiment: train
    exp_train = experiment.get_train()
    Engine.run_trainer(exp_train)

    # ---- Experiment: test
    exp_test = experiment.get_test()
    Engine.run_tester(exp_test)

# --------------------------------------------------------------
def parse_arguments():
    parser = argparse.ArgumentParser(description="", formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("-i", type=str, help="path to experiment config")
    args = parser.parse_args()
    return args.i

if __name__ == '__main__':
    path_cfg = parse_arguments()
    run(path_cfg)