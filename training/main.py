import hydra
from process import process_data
# from evaluate_model import evaluate
# from train_model import train

""" Call the config file """
@hydra.main(version_base=None, config_path="../config", config_name="main")
def main(config):
    process_data(config)
    # train(config)
    # evaluate(config)


if __name__ == "__main__":
    main()
