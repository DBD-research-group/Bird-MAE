# `BirdMAE`: A Bird Bioacoustic Foundation Model 

This repository hosts the code of "**Can Masked Autoencoders listen to Bird?**".  Masked Autoencoders (MAEs) pretrained on AudioSet fail to capture the fine-grained acoustic characteristics of specialized domains such as bioacoustic monitoring. Bird sound classification is critical for assessing environmental health, yet general-purpose models inadequately address its unique acoustic challenges. To address this, we introduce Bird-MAE, a domain-specialized MAE pretrained on the large-scale BirdSet dataset. We explore adjustments to pretraining, fine-tuning and utilizing frozen representations. Bird-MAE achieves state-of-the-art results across all BirdSet downstream tasks, substantially improving multi-label classification performance compared to the general-purpose Audio-MAE baseline. 

<br>
<div align="center">
  <img src="https://github.com/DBD-research-group/Bird-MAE/blob/main/docs/imgs/GA.png" alt="logo", width=600>
</div>
<br>

## Installation
You can install the environment to reproduce the results with: 

```
conda create -n birdmae python=3.10.14
pip  install -r requirements.txt
```

## Data Preparation
The paper includes three different types of data that is used: 

- `Pretraining data` for training the Bird-MAE model
- `Downstream data` for fine-tuning/probing on the complete downstream tasks from BirdSet
- `Few-shot data` for probing on few-shot setting in BirdSet

For each data type, you have to download and then prepare the data before running the experiments. You can also download the checkpoints of the respective model and skip the intensive pretraining. The downloading and preparation is available in the `util/prepare_data.py` file. 
Be sure to change the respective paths in the scripts. For each experiment, we add instruction on how to prepare the data. 

## Pretraining on `BirdSet`

### Data
You can run the script that downloads the `XCL` dataset from [Hugging Face](https://huggingface.co/datasets/DBD-research-group/BirdSet) and prepares it for pretraining to the curated `XCL-1.7M` from the terminal. Note that you need approximately 500 Gbs of disk space for the download (plus a little bit more for the prepared file).

```
python util/prepare_data/pretraining.py --save_path --cache_dir 
```

For example, if you want to get all events, without any curation: 

```
python util/prepare_data/pretraining.py \
    --dataset_name "XCL" \
    --hf_path "DBD-research-group/BirdSet" \
    --cache_dir "/data/birdset/XCL" \ # download directory of the dataset 
    --save_path "/data/birdset/XCL/XCL_processed_allevents" \ # sub directory of the save_to_disk path
    --class_limit 0 \
    --event_limit 0 \
    --audio_sampling_rate 32000 \
    --num_proc 1 \ # num proc during download
    --mapping_num_proc 4 # num proc during event mapping
```

### Experiments

The main pretraining script is `pretrain.py`. The experiments are managed by hydra. The pretraining configs for the base, large and huge model are available in `configs/experiment/paper/pretrain`. The respective slurm files can be found in `slurm/pretrain/{base,large,huge}`. You have to change the paths etc. Example: 
```
sbatch slurm/pretrain/large/large/large.sh
```

## Multi-Label Benchmark on `BirdSet`
### Data

You can use the script below to download and prepare the downstream BirdSet datasets for evaluation. It downloads each dataset from Hugging Face and caches it locally, then processes it into the required format for multi-label classification tasks.

Run the script from the terminal:

```bash
python util/prepare_data/downstream.py --cache-dir-base /data/birdset
```

You can also specify which datasets to process using the `--dataset-names` argument:

```bash
python util/prepare_data/downstream.py \
    --dataset-names PER NES HSN POW \
    --cache-dir-base /data/birdset
```

Each dataset is cached individually in a subfolder of `--cache-dir-base` and then prepared into a processed format used by the BirdSet pipeline.
Note that `classlimit`, `eventlimit`, and other parameters are defined inside the script and can be modified as needed.


### Experiments
All config files of the experiments for the multi-label benchmark (with fine-tuning and linear probing) are available in `configs/experiment/bigshot`. Example: 

``` bash
python train.py experiment="paper/bigshot/$model/$type/$head/$dataset"
```

## Multi-Label Few-Shot Benchmark on `BirdSet`
### Data 

You can use the script below to create k-shot BirdSet subsets for few-shot evaluation.
It first downloads each requested split from Hugging Face (if not cached), then samples the desired number of recordings per class under the chosen selection rule, and finally saves the processed `DatasetDict` to disk.

Run the script from the terminal:

```bash
python util/prepare_data/fewshot.py --cache-dir-base /data/birdset
```

You can customise the datasets, shot counts, and random seeds:

```bash
python util/prepare_data/fewshot.py \
    --dataset-names PER NES HSN POW \
    --shots 1 5 10 \
    --seeds 1 2 3 \
    --condition lenient \
    --cache-dir-base /data/birdset
```

Each subset is stored in a subfolder of `--cache-dir-base`, e.g.

```
/data/birdset/HSN/HSN_5shot_1/
```

The original **`test_5s`** split is kept intact, while the **`train`** split is replaced by the new k-shot set.
Sampling rules (`strict` vs `lenient`), `classlimit`, `eventlimit`, and other parameters are defined inside the script and can be adjusted as needed.


### Experiments
All config files of the experiments for the multi-label benchmark (with fine-tuning and linear probing) are available in `configs/experiment/fewshot`. Example: 

``` bash
python train.py experiment="paper/fewshot/$probing/$probing/$dataset_kshots"
```

## Checkpoints
- [Bird-MAE Base:](https://huggingface.co/anonymous-birder/Bird-MAE/blob/main/Bird-MAE-B)
- [Bird-MAE Large:](https://huggingface.co/anonymous-birder/Bird-MAE/blob/main/Bird-MAE-L)
- [Bird-MAE Huge:](https://huggingface.co/anonymous-birder/Bird-MAE/blob/main/Bird-MAE-H)
