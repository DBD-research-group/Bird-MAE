import os 
import hydra 
import torch
import lightning as L
import sys

from omegaconf import OmegaConf, DictConfig
import pyrootutils
from pathlib import Path 

from datamodule import HFDataModule, BirdSetDataModule
from util.pylogger import get_pylogger
from util.log_hparams import log_hyperparameters
from build_model import instantiate_callbacks, build_model

log = get_pylogger(__name__)

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git"],
    pythonpath=True,
    dotenv=True,
)

_HYDRA_PARAMS = {
    "version_base": None,
    "config_path": str(root / "configs"),
    "config_name": "finetune.yaml"
}

@hydra.main(**_HYDRA_PARAMS)
def finetune(cfg: DictConfig):
    log.info(f"Seed everything with {cfg.seed}")
    L.seed_everything(cfg.seed)
    #torch.set_num_threads(12)
    
    if "birdset" in cfg.data.dataset.hf_path.lower(): # correct this later 
        datamodule = BirdSetDataModule(
            dataset_configs=cfg.data.dataset,
            loader_configs=cfg.data.loaders,
            transform_configs=cfg.data.transform,
            sampling_rate=cfg.module.network.sampling_rate
        )

    if sys.gettrace():
         log.info("Debugging mode, no logger")
         logger = None
    else:
        log.info("Setup logger")
        logger = hydra.utils.instantiate(cfg.logger)

    log.info("Setup callbacks")
    callbacks = instantiate_callbacks(cfg["callbacks"])
                                      
    log.info("Setup trainer")
    trainer = L.Trainer(**cfg.trainer, callbacks=callbacks, logger=logger)

    log.info("Setup model")
    model = build_model(cfg.module)

    pretrained_weights_path = cfg.module.network.get("pretrained_weights_path", None)

    if pretrained_weights_path: 
        log.info(f"Load pretrained weights from {pretrained_weights_path}")
        model.load_pretrained_weights(pretrained_weights_path)

    if cfg.module.network.get("freeze_backbone", False): # move this to the models!
        log.info("Freezing backbone weights, only training head")
        if cfg.module.network.name == "ConvNext":
             for name, param in model.named_parameters():
                if 'classifier' not in name:
                    param.requires_grad = False
        else:
            for name, param in model.named_parameters():
                if 'head' not in name:
                    param.requires_grad = False
                
    object_dict = {
        "cfg": cfg, 
        "datamodule": datamodule,
        "model": model,
        "logger": logger,
        "trainer": trainer
    }

    if logger: 
        log.info("Logging hyperparameters")
        log_hyperparameters(object_dict)

    if cfg.train: 
        log.info("Start training")
        trainer.fit(model=model, datamodule=datamodule) 
                    #,ckpt_path="/home/anonymous/projects/birdMAE/logs/finetune/runs/audioset_balanced/VIT/2024-10-14_170415/model_checkpoints/last.ckpt")

    if cfg.test:
        log.info("Start testing")
        trainer.test(model=model, datamodule=datamodule, ckpt_path="last")

if __name__ == "__main__":
    finetune()


    

