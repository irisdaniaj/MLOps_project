import hydra
from omegaconf import DictConfig, OmegaConf

@hydra.main(config_path="mlops/data/Mydataset", config_name="config")
def main(cfg: DictConfig) -> None:
    print("Custom Data Config:")
    print(OmegaConf.to_yaml(cfg.customdata))
    print("\nHyperparameters Config:")
    print(OmegaConf.to_yaml(cfg.customconfig))

if __name__ == "__main__":
    main()

import hydra
from omegaconf import DictConfig, OmegaConf

@hydra.main(config_path="mlops/data/Mydataset", config_name="customdata")
def customdata(cfg: DictConfig) -> None:
    print("Customdata:")
    print(OmegaConf.to_yaml(cfg))

@hydra.main(config_path="mlops/data/Mydataset", config_name="customconfig")
def customconfig(cfg: DictConfig) -> None:
    print("Customconfig:")
    print(OmegaConf.to_yaml(cfg))

if __name__ == "__main__":
    customdata()
    customconfig()