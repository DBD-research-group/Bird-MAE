import os 
import json
from datetime import datetime
from omegaconf import DictConfig
from datasets import load_dataset, Audio, ClassLabel, load_from_disk, Sequence
from torch.utils.data import DataLoader
import lightning.pytorch as pl 
from lightning.pytorch.utilities.rank_zero import rank_zero_info
from lightning.pytorch.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
import numpy as np 
from transforms import TrainTransform, EvalTransform, ImageTrainTransform, ImageEvalTransform, BirdSetTrainTransform

class HFDataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset_configs: DictConfig,
        loader_configs: DictConfig,
        transform_configs: DictConfig,
        sampling_rate: int
        ):

        super().__init__()

        self.hf_path = dataset_configs.hf_path
        self.hf_name = dataset_configs.hf_name
        self.data_dir = dataset_configs.dataset_dir
        self.num_classes = dataset_configs.num_classes
        self.test_size = dataset_configs.test_size
        self.train_split = dataset_configs.train_split
        self.test_split = dataset_configs.test_split
        self.num_workers = dataset_configs.num_workers
        self.columns = dataset_configs.columns
        self.sampling_rate = sampling_rate
        self.save_to_disk = dataset_configs.save_to_disk
        self.test_in_val = dataset_configs.test_in_val
        self.saved_images = dataset_configs.saved_images

        self.train_image_transform = ImageTrainTransform(
            transform_params=transform_configs,
            sampling_rate=sampling_rate,
            target_length=dataset_configs.target_length,
            mean=dataset_configs.mean,
            std=dataset_configs.std,
            columns = dataset_configs.columns,
            clip_duration = dataset_configs.clip_duration
        )

        self.val_image_transform = ImageEvalTransform(
            transform_params=transform_configs,
            sampling_rate=sampling_rate,
            target_length=dataset_configs.target_length,
            mean=dataset_configs.mean,
            std=dataset_configs.std,
            columns = dataset_configs.columns,
            clip_duration = dataset_configs.clip_duration
        )

        self.test_image_transform = ImageEvalTransform(
            transform_params=transform_configs,
            sampling_rate=sampling_rate,
            target_length=dataset_configs.target_length,
            mean=dataset_configs.mean,
            std=dataset_configs.std,
            columns = dataset_configs.columns,
            clip_duration = dataset_configs.clip_duration
        )

        self.train_transform = TrainTransform(
            transform_params=transform_configs,
            sampling_rate=sampling_rate,
            target_length=dataset_configs.target_length,
            mean=dataset_configs.mean,
            std=dataset_configs.std,
            columns = dataset_configs.columns,
            clip_duration = dataset_configs.clip_duration
        )

        self.val_transform = EvalTransform(
            transform_params=transform_configs,
            sampling_rate=sampling_rate,
            target_length=dataset_configs.target_length,
            mean=dataset_configs.mean,
            std=dataset_configs.std,
            columns = dataset_configs.columns,
            clip_duration = dataset_configs.clip_duration
        )

        self.test_transform = EvalTransform(
            transform_params=transform_configs,
            sampling_rate=sampling_rate,
            target_length=dataset_configs.target_length,
            mean=dataset_configs.mean,
            std=dataset_configs.std,
            columns = dataset_configs.columns,
            clip_duration = dataset_configs.clip_duration
        )

        self.train_loader_configs = loader_configs.train
        self.val_loader_configs = loader_configs.val
        self.test_loader_configs = loader_configs.test

    def prepare_data(self):
        #pl.seed_everything(self.seed) ## needed? 
        rank_zero_info(">> Preparing data")
        if not os.path.exists(self.data_dir):
            rank_zero_info(f"[{str(datetime.now())}] Data directory {self.data_dir} does not exist. Creating it.")
            os.makedirs(self.data_dir)
        
        cache_dir_is_empty = len(os.listdir(self.data_dir)) == 0
        
        if cache_dir_is_empty:
            rank_zero_info(f"[{str(datetime.now())}] Downloading dataset.")
            if self.hf_name:
                load_dataset(self.hf_path, self.hf_name, cache_dir=self.data_dir, load_from_cache_file=True)
            else:
                load_dataset(self.hf_path, cache_dir=self.data_dir)
        else:
            rank_zero_info(
                f"[{str(datetime.now())}] Data cache {self.data_dir} exists. Loading from cache in setup."
            )
    
    def setup(self, stage:str) -> None: 
        if stage == "fit" or stage is None: 
            if self.saved_images:
                self.train_data = load_from_disk(f"{self.saved_images}/train")
                self.train_data.set_transform(self.train_image_transform)

                try: 
                    self.val_data = load_from_disk(f"{self.saved_images}/test")
                    self.val_data.set_transform(self.val_image_transform) # improve!!
                except: 
                    print("no test in saved images")
                    
            else:
                if self.save_to_disk: 
                    dataset = load_from_disk(f"{self.save_to_disk}/train")
                else: 
                    dataset = load_dataset(
                        self.hf_path, self.hf_name, split=self.train_split, cache_dir=self.data_dir
                    )

                    if "AudioSet" in self.hf_path:
                        with open("/home/anonymous/projects/birdMAE/data/audioset_ontology_custom527.json", "r") as f:
                            ontology = json.load(f)
                        num_classes = len(ontology)
                        label_names = list(ontology.keys())
                        class_label = Sequence(ClassLabel(num_classes=num_classes, names=label_names))
                        dataset = dataset.cast_column("human_labels", class_label)
                        dataset = dataset.map(self._one_hot_encode, batched=True, batch_size=1000, load_from_cache_file=True)

                        rows_to_remove = [15_759,17_532] #corrupted
                        all_indices = list(range(len(dataset)))
                        indices_to_keep = [i for i in all_indices if i not in rows_to_remove]
                        dataset = dataset.select(indices_to_keep)


                if self.test_size:
                    split = dataset.train_test_split(
                        self.test_size,
                        shuffle=True,
                        seed=42
                    )
                    self.train_data = split["train"]
                    self.val_data = split["test"]
                
                else: 
                    self.train_data = dataset
                    self.val_data = None

                self.train_data.set_format("numpy", columns=self.columns, output_all_columns=False)
                self.train_data = self.train_data.cast_column("audio", Audio(sampling_rate=self.sampling_rate, mono=True, decode=True))
                self.train_data.set_transform(self.train_transform)
                #self.train_data = self.train_data.select(range(100))

                if self.val_data:
                    self.val_data.set_format("numpy", columns=self.columns, output_all_columns=False)
                    self.val_data = self.val_data.cast_column("audio", Audio(sampling_rate=self.sampling_rate, mono=True, decode=True))
                    self.val_data.set_transform(self.val_transform)
                
                if self.test_in_val == True: # not nice, only for as
                    if self.save_to_disk: 
                        self.val_data = load_from_disk(f"{self.save_to_disk}/test")
                    else: 
                        self.val_data = load_dataset(self.hf_path, self.hf_name, split=self.test_split, cache_dir=self.data_dir)

                    self.val_data.set_format("numpy", columns=self.columns, output_all_columns=False)
                    self.val_data = self.val_data.cast_column("audio", Audio(sampling_rate=self.sampling_rate, mono=True, decode=True))
                    self.val_data.set_transform(self.val_transform)

        
        if stage == "test": 
            if self.saved_images:
                self.test_data = load_from_disk(f"{self.saved_images}/test")
                self.test_data.set_transform(self.test_image_transform)
            
            else:
                if self.save_to_disk: 
                    self.test_data = load_from_disk(f"{self.save_to_disk}/test")
                else: 
                    self.test_data = load_dataset(self.hf_path, self.hf_name, split=self.test_split, cache_dir=self.data_dir)

                    if "AudioSet" in self.hf_path:
                        with open("/home/anonymous/projects/birdMAE/data/audioset_ontology_custom527.json", "r") as f:
                            ontology = json.load(f)
                        num_classes = len(ontology)
                        label_names = list(ontology.keys())
                        class_label = Sequence(ClassLabel(num_classes=num_classes, names=label_names))
                        self.test_data = self.test_data.cast_column("human_labels", class_label)
                        self.test_data = self.test_data.map(self._one_hot_encode, batched=True, batch_size=1000)

                        rows_to_remove = [6_182] #corrupted
                        all_indices = list(range(len(self.test_data)))
                        indices_to_keep = [i for i in all_indices if i not in rows_to_remove]
                        self.test_data = self.test_data.select(indices_to_keep)

                self.test_data.set_format("numpy", columns=self.columns, output_all_columns=False)
                self.test_data = self.test_data.cast_column("audio", Audio(sampling_rate=self.sampling_rate, mono=True, decode=True))
                self.test_data.set_transform(self.test_transform)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_data,
            num_workers=self.train_loader_configs.num_workers,
            batch_size=self.train_loader_configs.batch_size,
            shuffle=self.train_loader_configs.shuffle
        )
    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_data,
            num_workers=self.val_loader_configs.num_workers,
            batch_size=self.val_loader_configs.batch_size,
            shuffle=self.val_loader_configs.shuffle
        )
    
    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_data,
            num_workers=self.test_loader_configs.num_workers,
            batch_size=self.test_loader_configs.batch_size,
            shuffle=self.test_loader_configs.shuffle
        )
    
    def _one_hot_encode(self, batch):
        label_list = [y for y in batch[self.columns[1]]]
        
        # Use numpy instead of torch for caching
        class_one_hot_matrix = np.zeros((len(label_list), self.num_classes), dtype=np.float32)
        
        for class_idx, indices in enumerate(label_list):
            class_one_hot_matrix[class_idx, indices] = 1.0
        
        return {self.columns[1]: class_one_hot_matrix}
        
class BirdSetDataModule(HFDataModule):
    def __init__(
            self, 
            dataset_configs: DictConfig, 
            loader_configs: DictConfig, 
            transform_configs: DictConfig, 
            sampling_rate: int
            ):
        super().__init__(
            dataset_configs, 
            loader_configs, 
            transform_configs, 
            sampling_rate)
        
        self.train_transform = BirdSetTrainTransform(
            transform_params=transform_configs,
            sampling_rate=sampling_rate,
            target_length=dataset_configs.target_length,
            mean=dataset_configs.mean,
            std=dataset_configs.std,
            columns = dataset_configs.columns,
            clip_duration = dataset_configs.clip_duration)
        
        self.val_transform = EvalTransform(
            transform_params=transform_configs,
            sampling_rate=sampling_rate,
            target_length=dataset_configs.target_length,
            mean=dataset_configs.mean,
            std=dataset_configs.std,
            columns = dataset_configs.columns,
            clip_duration = dataset_configs.clip_duration
        )

        self.test_transform = EvalTransform(
            transform_params=transform_configs,
            sampling_rate=sampling_rate,
            target_length=dataset_configs.target_length,
            mean=dataset_configs.mean,
            std=dataset_configs.std,
            columns = dataset_configs.columns,
            clip_duration = dataset_configs.clip_duration
        )

        self.train_loader_configs = loader_configs.train
        self.val_loader_configs = loader_configs.val
        self.test_loader_configs = loader_configs.test

    def setup(self, stage:str) -> None: 
        if stage == "fit" or stage is None: 
            if self.saved_images:
                self.train_data = load_from_disk(f"{self.saved_images}/train")
                self.train_data.set_transform(self.train_image_transform)
                self.val_data = None

                try: 
                    self.val_data = load_from_disk(f"{self.saved_images}/test")
                    self.val_data.set_transform(self.val_image_transform) # improve!!
                except: 
                    print("no test in saved images")
            else:
                self.train_data = load_from_disk(f"{self.save_to_disk}/train")
                self.val_data = None

                if self.hf_name != "XCM" and self.hf_name != "XCL":
                    print("no val in xc")
                    try: 
                        self.val_data = load_from_disk(f"{self.save_to_disk}/valid")
                    except: 
                        print("no valid in data dir")
                        self.val_data = None
            
                self.train_data.set_format("numpy", columns=self.columns, output_all_columns=False)
                #self.train_data = self.train_data.cast_column("audio", Audio(sampling_rate=self.sampling_rate, mono=True, decode=True))
                self.train_data.set_transform(self.train_transform)
                #self.train_data = self.train_data.select(range(100))

                if self.val_data:
                    self.val_data.set_format("numpy", columns=self.columns, output_all_columns=False)
                    #self.val_data = self.val_data.cast_column("audio", Audio(sampling_rate=self.sampling_rate, mono=True, decode=True))
                    self.val_data.set_transform(self.val_transform)
                    
                if self.test_in_val == True: # not nice, only for as
                    self.val_data = load_from_disk(f"{self.save_to_disk}/test")
                    self.val_data.set_format("numpy", columns=self.columns, output_all_columns=False)
                    self.val_data.set_transform(self.val_transform)

        
        if stage == "test": 
            self.test_data = load_from_disk(f"{self.save_to_disk}/test")

            self.test_data.set_format("numpy", columns=self.columns, output_all_columns=False)
            self.test_data.set_transform(self.test_transform)


from util.mask_jepa import apply_masks, MaskCollator


class BirdSetDataModule_JEPA(HFDataModule):
    def __init__(
            self, 
            dataset_configs: DictConfig, 
            loader_configs: DictConfig, 
            transform_configs: DictConfig, 
            sampling_rate: int
            ):
        super().__init__(
            dataset_configs, 
            loader_configs, 
            transform_configs, 
            sampling_rate)
        
        self.train_transform = BirdSetTrainTransform(
            transform_params=transform_configs,
            sampling_rate=sampling_rate,
            target_length=dataset_configs.target_length,
            mean=dataset_configs.mean,
            std=dataset_configs.std,
            columns = dataset_configs.columns,
            clip_duration = dataset_configs.clip_duration)
        
        self.val_transform = EvalTransform(
            transform_params=transform_configs,
            sampling_rate=sampling_rate,
            target_length=dataset_configs.target_length,
            mean=dataset_configs.mean,
            std=dataset_configs.std,
            columns = dataset_configs.columns,
            clip_duration = dataset_configs.clip_duration
        )

        self.test_transform = EvalTransform(
            transform_params=transform_configs,
            sampling_rate=sampling_rate,
            target_length=dataset_configs.target_length,
            mean=dataset_configs.mean,
            std=dataset_configs.std,
            columns = dataset_configs.columns,
            clip_duration = dataset_configs.clip_duration
        )

        mask_configs = transform_configs.mask_configs
        self.mask_collator = MaskCollator(
            input_size=mask_configs.input_size,
            patch_size=mask_configs.patch_size,
            pred_mask_scale=mask_configs.pred_mask_scale,
            enc_mask_scale=mask_configs.enc_mask_scale,
            aspect_ratio=mask_configs.aspect_ratio,
            aspect_ratio_context=mask_configs.aspect_ratio_context,
            nenc=mask_configs.nenc,
            npred=mask_configs.npred,
            min_keep=mask_configs.min_keep,
            allow_overlap=mask_configs.allow_overlap,
        )

        self.train_loader_configs = loader_configs.train
        self.val_loader_configs = loader_configs.val
        self.test_loader_configs = loader_configs.test

    def setup(self, stage:str) -> None: 
        if stage == "fit" or stage is None: 
            if self.saved_images:
                self.train_data = load_from_disk(f"{self.saved_images}/train")
                self.train_data.set_transform(self.train_image_transform)
                self.val_data = None

                try: 
                    self.val_data = load_from_disk(f"{self.saved_images}/test")
                    self.val_data.set_transform(self.val_image_transform) # improve!!
                except: 
                    print("no test in saved images")
            else:
                self.train_data = load_from_disk(f"{self.save_to_disk}/train")
                self.val_data = None

                if self.hf_name != "XCM" and self.hf_name != "XCL":
                    print("no val in xc")
                    self.val_data = load_from_disk(f"{self.save_to_disk}/valid")
            
                self.train_data.set_format("numpy", columns=self.columns, output_all_columns=False)
                #self.train_data = self.train_data.cast_column("audio", Audio(sampling_rate=self.sampling_rate, mono=True, decode=True))
                self.train_data.set_transform(self.train_transform)
                #self.train_data = self.train_data.select(range(100))

                if self.val_data:
                    self.val_data.set_format("numpy", columns=self.columns, output_all_columns=False)
                    #self.val_data = self.val_data.cast_column("audio", Audio(sampling_rate=self.sampling_rate, mono=True, decode=True))
                    self.val_data.set_transform(self.val_transform)
                    
                if self.test_in_val == True: # not nice, only for as
                    self.val_data = load_from_disk(f"{self.save_to_disk}/test")
                    self.val_data.set_format("numpy", columns=self.columns, output_all_columns=False)
                    self.val_data.set_transform(self.val_transform)

        
        if stage == "test": 
            self.test_data = load_from_disk(f"{self.save_to_disk}/test")

            self.test_data.set_format("numpy", columns=self.columns, output_all_columns=False)
            self.test_data.set_transform(self.test_transform)
    
    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_data,
            num_workers=self.train_loader_configs.num_workers,
            batch_size=self.train_loader_configs.batch_size,
            shuffle=self.train_loader_configs.shuffle,
            drop_last=True,
            collate_fn=self.mask_collator
        )
    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_data,
            num_workers=self.val_loader_configs.num_workers,
            batch_size=self.val_loader_configs.batch_size,
            shuffle=self.val_loader_configs.shuffle
        )
    
    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_data,
            num_workers=self.test_loader_configs.num_workers,
            batch_size=self.test_loader_configs.batch_size,
            shuffle=self.test_loader_configs.shuffle
        )