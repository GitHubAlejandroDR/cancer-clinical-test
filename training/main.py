import hydra
from evaluate_model import evaluate
from process import process_data
from train_model import train

"""Call the config file"""


@hydra.main(version_base=None, config_path="../config", config_name="main")
def main(config):
    """
    Execute the main function using the specified configuration.

    This function is the entry point of the program. It uses the Hydra framework to load
    the main configuration file and executes the 'process_data' function.

    Args:
        config: The loaded configuration object.

    Returns:
        None
    """
    process_data(config)
    # train(config)
    # evaluate(config)


if __name__ == "__main__":
    main()
