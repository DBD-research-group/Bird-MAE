
#%%

from datasets import load_dataset, Audio

dataset = load_dataset("marsyas/gtzan", "all", trust_remote_code=True, cache_dir="./data/gtzan")
dataset = dataset.cast_column("audio", Audio(sampling_rate=32000))

# already waveforms! 
#%%
dataset
#%%
from tqdm import tqdm 

for i in tqdm(range(len(dataset["train"]))):
    dataset["train"][i]["audio"]["array"][0]



#dataset["train"][0]["file"]
#%%
dataset["train"][0]["audio"]


#%%
dataset_size_bytes  = dataset["train"].dataset_size
dataset_size_mb = dataset_size_bytes / (1024 ** 2)
dataset_size_gb = dataset_size_bytes / (1024 ** 3)

print(f"Dataset size: {dataset_size_bytes} bytes")
print(f"Dataset size: {dataset_size_mb:.2f} MB")
print(f"Dataset size: {dataset_size_gb:.2f} GB")
#%%


from datasets import load_dataset, Audio

dataset = load_dataset("PolyAI/minds14", "en-US", split="train", trust_remote_code=True, cache_dir="./data/minds14")
dataset = dataset.cast_column("audio", Audio(sampling_rate=32_000)) # already waveforms! 

#%%

from transformers import AutoFeatureExtractor

model_id = "ntu-spml/distilhubert"
feature_extractor = AutoFeatureExtractor.from_pretrained(
    model_id, do_normalize=True, return_attention_mask=True
)
#%%

sampling_rate = feature_extractor.sampling_rate
sampling_rate = 32_000
#%%

from datasets import Audio

gtzan = gtzan.cast_column("audio", Audio(sampling_rate=sampling_rate))

#%%
from datasets import load_from_disk
from datasets import Audio

dataset = load_from_disk("/home/anonymous/projects/birdMAE/data/HSN/HSN_processed_42_cdb073221fc18e3d")

dataset
#%%
dataset = dataset.cast_column("filepath", Audio(sampling_rate=32_000, decode=True, mono=True))

#%%
dataset["train"][0]["filepath"]


#%%
from tqdm import tqdm 

for i in tqdm(range(100)):
    dataset["train"][i]["filepath"]["array"]

#%%

small = dataset["train"].select(range(100))
#%%
small

#%%
from torchaudio.compliance.kaldi import fbank
import torch
fbank_features_test = fbank(
                torch.from_numpy(testarray).unsqueeze(0),
                htk_compat=True,
                sample_frequency=32_000,
                use_energy=False,
                window_type='hanning',
                num_mel_bins=128,
                dither=0.0,
                frame_shift=10
)
#%%

def prepare_dataset(batch):
    audio = batch['filepath']
    batch["input_values"] = audio["array"][:160_000]

    batch["label"] = batch["labels"]
    return batch


# Apply the mapping function to the dataset
mapped_dataset = small.map(
    prepare_dataset, 
    remove_columns=small.column_names)
#%%
small[0]["filepath"]["array"]
#%%
from torchaudio.compliance.kaldi import fbank
import torch 
import numpy as np
from torch.nn.functional import pad
from PIL import Image

def prepare_dataset(batch):
    #pad_to_160k = lambda x: pad(x, (0, 160_000 - x.shape[0]), "constant", 0)
    #data = [torch.from_numpy(b["array"][:160_000]) for b in batch["audio"]]
    data = [torch.from_numpy(b["array"]) for b in batch["audio"]]
    #data = [pad_to_160k(d) for d in data]

    imgs = []
    for d in data: 
        img = fbank(
            d.unsqueeze(0),
            htk_compat=True,
            sample_frequency=32_000,
            use_energy=False,
            window_type='hanning',
            num_mel_bins=128,
            dither=0.0,
            frame_shift=10
        )
        imgs.append(img.T)
    imgs = [Image.fromarray(img.numpy()) for img in imgs]
    
    # del batch['filepath']
    # del batch['detected_events']
    # del batch['start_time']
    # del batch['end_time']
    batch['input_values'] = imgs
    batch["label"] = batch["human_labels"]

    return batch

#%%

from datasets import Audio, load_dataset
import json
from datasets import Sequence, ClassLabel
dataset = load_dataset(
    "agkphysics/AudioSet", 
    cache_dir="/home/anonymous/projects/birdMAE/data/audioset_balanced")

dataset = dataset.cast_column("audio", Audio(sampling_rate=32_000))
def _one_hot_encode(batch):
    label_list = [y for y in batch["human_labels"]]
    
    # Use numpy instead of torch for caching
    class_one_hot_matrix = np.zeros((len(label_list), 527), dtype=np.float32)
    
    for class_idx, indices in enumerate(label_list):
        class_one_hot_matrix[class_idx, indices] = 1.0
    
    return {"human_labels": class_one_hot_matrix}
with open("/home/anonymous/projects/birdMAE/data/audioset_ontology_custom527.json", "r") as f:
    ontology = json.load(f)
num_classes = len(ontology)
label_names = list(ontology.keys())
class_label = Sequence(ClassLabel(num_classes=num_classes, names=label_names))
dataset = dataset.cast_column("human_labels", class_label)
dataset = dataset.map(_one_hot_encode, batched=True, batch_size=1000, load_from_cache_file=True)

rows_to_remove = [15_759,17_532] #corrupted
all_indices = list(range(len(dataset["train"])))
indices_to_keep = [i for i in all_indices if i not in rows_to_remove]
dataset["train"] = dataset["train"].select(indices_to_keep)

rows_to_remove = [6_182] #corrupted
all_indices = list(range(len(dataset["test"])))
indices_to_keep = [i for i in all_indices if i not in rows_to_remove]
dataset["test"] = dataset["test"].select(indices_to_keep)

#%%

dataset["train"][0]["human_labels"]
#%%

dataset

#%%

from tqdm import tqdm 
for i in tqdm(range(len(dataset["train"]))):
    dataset["train"][i]["audio"]["array"]
#%%


dataset= dataset.map(
    prepare_dataset,
    remove_columns=dataset["train"].column_names,
    batched=True,
    batch_size=100)

#%%

dataset["train"][0]["label"]
#%%

dataset.save_to_disk("./data/audioset_balanced_prepared")

#%%
from datasets import load_dataset, load_from_disk

dataset = load_from_disk("./data/audioset_balanced_prepared")
#%%
import pylab as plt
import numpy as np
plt.imshow(np.array(dataset["train"][0]["input_values"]), cmap='viridis')

#%%
from datasets import Audio, load_dataset

dataset_ = load_dataset(
    "agkphysics/AudioSet", 
    cache_dir="/home/anonymous/projects/birdMAE/data/audioset_balanced")

dataset_ = dataset_.cast_column("audio", Audio(sampling_rate=32_000))
from torchaudio.compliance.kaldi import fbank
import torch
fbank_features_test = fbank(
                torch.from_numpy(dataset_["train"][0]["audio"]["array"]).unsqueeze(0),
                htk_compat=True,
                sample_frequency=32_000,
                use_energy=False,
                window_type='hanning',
                num_mel_bins=128,
                dither=0.0,
                frame_shift=10
)
#%%
fbank_features_test.shape

plt.imshow(fbank_features_test.T, cmap='viridis')

#%%

fbank_features_test.T - np.array(dataset["train"][0]["input_values"])
#%%
import matplotlib.pyplot as plt
from tqdm import tqdm 
for i in tqdm(range(len(dataset["train"]))):
    plt.imshow(np.array(dataset["train"][i]["input_values"]), cmap='viridis')
    plt.show()
#%%

np.array(dataset["train"][0]["input_values"])



#%%

from datasets import Audio, load_dataset

dataset = load_dataset(
    "ashraq/esc50", 
    cache_dir="./data/esc50",
)
dataset = dataset.cast_column("audio", Audio(sampling_rate=32_000))
#%%
from tqdm import tqdm 
for i in tqdm(range(len(dataset["train"]))):
    dataset["train"][i]["audio"]["array"]

#%%
#%%

dataset

#%%
dataset= dataset.map(
    prepare_dataset,
    remove_columns=dataset["train"].column_names,
    batched=True,
    batch_size=500)
#%%
dataset

#%%
from tqdm import tqdm 
for i in tqdm(range(len(dataset["train"]))):
    np.array(dataset["train"][i]["input_values"])
#%%
dataset.save_to_disk("./data/esc50_prepared_n")
#%%

dataset["train"][10]["input_values"]

#%%






dataset = load_dataset(
    "ashraq/esc50", 
    cache_dir="./data/esc50",
    split="train",
)
dataset = dataset.cast_column("audio", Audio(sampling_rate=32_000))
dataset_small = dataset.select(range(100))
dataset_smaller = dataset_small.map(
    prepare_dataset,
    remove_columns=dataset_small.column_names,
    batched=True,
    batch_size=2)
plt.imshow(np.array(dataset_smaller[0]["input_values"]), cmap='viridis')

#%%

# Apply the mapping function to the dataset
mapped_dataset = small.map(
    prepare_dataset,
    remove_columns=small.column_names,
    batched=True,
    batch_size=1)

#%%
import pylab as plt
from torchvision.utils import make_grid
rnd_idx = np.random.randint(0, len(mapped_dataset))
sample = mapped_dataset[rnd_idx]
print(sample)


img = sample['input_values']
# img_max = sample['max']
# img_min = sample['min']
# print(type(img))
# original_img = (img  * (img_max - img_min)) + img_min
# print(original_img.min(), original_img.max())
print(img.min(), img.max())
plt.imshow(img, cmap='viridis')
plt.show()

#%%
from datasets import load_dataset

dataset = load_dataset(
    "ashraq/esc50", 
    cache_dir="./data/esc50",
    split="train",
)
dataset = dataset.cast_column("audio", Audio(sampling_rate=32_000))


#%%

dataset_small = dataset.select(range(100))
#%%

len(dataset_small[0]["audio"]["array"])

#%%

dataset_smaller = dataset_small.map(
    prepare_dataset,
    remove_columns=dataset_small.column_names,
    batched=True,
    batch_size=1)
#%%

dataset_smaller[0]
#%%
np.array(dataset_smaller[0]["input_values"])
#%%
plt.imshow(dataset_smaller[0]["input_values"], cmap='viridis')

#%%



mapped_dataset[:2]



#%%
mapped_dataset[0]["input_values"].shape
#%%
from tqdm import tqdm 

for i in tqdm(range(100)):
    mapped_dataset[i]["filepath"]["array"]
#%%
mapped_dataset._get_cache_file_path("input_values")
#%%
#%%
dataset_size_bytes  = dataset["train"].dataset_size
dataset_size_mb = dataset_size_bytes / (1024 ** 2)
dataset_size_gb = dataset_size_bytes / (1024 ** 3)

print(f"Dataset size: {dataset_size_bytes} bytes")
print(f"Dataset size: {dataset_size_mb:.2f} MB")
print(f"Dataset size: {dataset_size_gb:.2f} GB")
#dataset = dataset.cast_column("audio", Audio(sampling_rate=32_000))

#%%
from datasets import load_dataset

dataset = load_dataset("DBD-research-group/BirdSet", "XCM", cache_dir="/home/anonymous/projects/birdMAE/data/XCM", num_proc=3)


#%%

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

num_layers = 25  # 24 blocks + classification head
layer_decay = 0.5

def get_layer_scales(num_layers, layer_decay, decay_type="right"):
    if decay_type == "right":
        # For layers 0 to num_layers-1, then append classification head scale 1
        scales = [layer_decay ** (num_layers - i) for i in range(num_layers)]
        scales.append(1.0)
    elif decay_type == "normal":
        # Use a normalized normal (Gaussian) PDF over the layers
        x = np.linspace(-3, 3, num_layers + 1)
        pdf_vals = norm.pdf(x)
        scales = (pdf_vals / np.max(pdf_vals)).tolist()
    elif decay_type == "inverse_normal":
        # Create an inverse-normal decay, adjusting the center
        x = np.linspace(-3, 3, num_layers + 1)
        pdf_vals = norm.pdf(x)
        pdf_vals = pdf_vals / np.max(pdf_vals)
        inverted = 1 - pdf_vals
        midpoint = len(inverted) // 2  # 26 // 2 = 13
        # Compute scaling factors for the left part (13 values)
        position_counts = np.arange(1, num_layers + 1)[::-1]  # shape: (25,)
        scaling_factors = layer_decay ** position_counts
        scaled_left = scaling_factors[:midpoint]  # first 13 values
        # Use the right part from the inverted PDF starting at midpoint (13 values)
        right = inverted[midpoint:]
        # Adjust a couple of values for a smoother transition if needed
        if len(right) >= 2:
            right[0] += 0.1
            right[1] += 0.1
        # Concatenate so that total length is 13 + 13 = 26
        scales = np.concatenate([scaled_left, right]).tolist()
    else:
        raise ValueError("Unknown decay_type: {}".format(decay_type))
    return scales

# Define the decay types we want to visualize
decay_types = ["right", "normal", "inverse_normal"]

# Create subplots: one for each decay type
fig, axs = plt.subplots(1, 3, figsize=(18, 5))

for ax, dt in zip(axs, decay_types):
    scales = get_layer_scales(num_layers, layer_decay, dt)
    # Both x and scales have 26 elements now
    ax.plot(range(num_layers + 1), scales, marker='o')
    ax.set_title(f"Decay type: {dt}")
    ax.set_xlabel("Layer Index (0 to 24, classification head at 25)")
    ax.set_ylabel("Learning Rate Scale")
    ax.grid(True)

plt.tight_layout()
plt.show()

#%%

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

num_layers = 25  # 24 blocks + classification head
layer_decay = 0.75  # baseline scale at the boundaries

def get_layer_scales(num_layers, layer_decay, decay_type="improved_inverse_normal", **kwargs):
    """
    Returns a list of learning rate scales for layers 0 to num_layers-1 plus one for the classification head.
    
    Available decay types:
      - "right": Exponential decay: scale_i = layer_decay^(num_layers - i)
      - "normal": Normal (Gaussian) PDF normalized between 0 and 1.
      - "inverse_normal": Original inverse-normal decay (less smooth).
      - "improved_inverse_normal": A smooth bump curve that keeps medium layers nearly unchanged.
      - "flat_middle": Piecewise decay with a flat region in the middle.
      - "polynomial": Polynomial decay: 1 - (1 - layer_decay)*((num_layers - i)/num_layers)^p
    """
    if decay_type == "right":
        scales = [layer_decay ** (num_layers - i) for i in range(num_layers)]
        scales.append(1.0)
    
    elif decay_type == "normal":
        x = np.linspace(-3, 3, num_layers + 1)
        pdf_vals = norm.pdf(x)
        scales = (pdf_vals / np.max(pdf_vals)).tolist()
    
    elif decay_type == "inverse_normal":
        # Original (ugly) inverse-normal version for reference
        x = np.linspace(-3, 3, num_layers + 1)
        pdf_vals = norm.pdf(x)
        pdf_vals = pdf_vals / np.max(pdf_vals)
        inverted = 1 - pdf_vals
        midpoint = len(inverted) // 2
        position_counts = np.arange(1, num_layers + 1)[::-1]
        scaling_factors = layer_decay ** position_counts
        scaled_left = scaling_factors[:midpoint]
        right = inverted[midpoint:]
        if len(right) >= 2:
            right[0] += 0.1
            right[1] += 0.1
        scales = np.concatenate([scaled_left, right]).tolist()
    
    elif decay_type == "improved_inverse_normal":
        # Improved version: create a smooth Gaussian bump centered in the middle.
        # Parameter sigma controls the width of the bump (default 0.3)
        sigma = kwargs.get("sigma", 0.3)
        # Create a normalized index x in [0,1] for layers (excluding the classification head)
        x = np.linspace(0, 1, num_layers)
        # Gaussian bump centered at 0.5, so that the middle layers get a scale of 1.
        bump = np.exp(-((x - 0.5) ** 2) / (2 * sigma ** 2))
        bump = bump / np.max(bump)  # now max is 1
        # Interpolate between layer_decay (at boundaries) and 1 (at the center)
        scales = layer_decay + (1 - layer_decay) * bump
        scales = scales.tolist()
        scales.append(1.0)
    
    elif decay_type == "flat_middle":
        flat_width = kwargs.get("flat_width", 4)
        L = num_layers  # excluding classification head
        mid = L // 2
        scales = []
        for i in range(L):
            if i < mid - flat_width // 2:
                scale = layer_decay ** (mid - i)
            elif i > mid + flat_width // 2:
                scale = layer_decay ** (i - mid)
            else:
                scale = 1.0
            scales.append(scale)
        scales.append(1.0)
    
    elif decay_type == "polynomial":
        power = kwargs.get("power", 2)
        scales = [1 - (1 - layer_decay) * ((num_layers - i) / num_layers) ** power for i in range(num_layers)]
        scales.append(1.0)
    
    else:
        raise ValueError("Unknown decay_type: {}".format(decay_type))
    
    return scales

# Define decay types to visualize
decay_configs = {
    "right": {},
    "normal": {},
    "inverse_normal": {},
    "improved_inverse_normal": {"sigma": 0.3},
    "flat_middle": {"flat_width": 4},
    "polynomial": {"power": 2},
}

# Plot all decay curves for comparison
num_configs = len(decay_configs)
fig, axs = plt.subplots(2, 3, figsize=(18, 10), sharey=True)
axs = axs.flatten()

for ax, (dt, params) in zip(axs, decay_configs.items()):
    scales = get_layer_scales(num_layers, layer_decay, dt, **params)
    ax.plot(range(num_layers + 1), scales, marker='o')
    ax.set_title(f"Decay: {dt}")
    ax.set_xlabel("Layer Index (0 to 24, classification head at 25)")
    ax.grid(True)
    
axs[0].set_ylabel("Learning Rate Scale")
plt.tight_layout()
plt.show()

#%%

import numpy as np
import matplotlib.pyplot as plt

num_layers = 25  # 24 blocks + classification head
layer_decay = 0.75

def get_layer_scales(num_layers, layer_decay, decay_type="right_modified", **kwargs):
    if decay_type == "right_modified":
        # Use a nonlinear mapping for the exponent.
        # beta < 1 will push the medium layers to have an even lower scale.
        beta = kwargs.get("beta", 0.9)
        scales = []
        # i from 0 to num_layers-1 for the blocks
        for i in range(num_layers):
            # Normalize layer index: x goes from 1 (earliest) to 0 (last block)
            x = 1 - (i / (num_layers - 1))
            # Instead of linear (num_layers - i), use a nonlinear mapping:
            exponent = 1 + (num_layers - 1) * (x ** beta)
            scale = layer_decay ** exponent
            scales.append(scale)
        scales.append(1.0)  # classification head fixed at 1
        return scales
    else:
        raise ValueError("Unknown decay_type: {}".format(decay_type))

# For comparison, here's the original right decay for reference:
def get_layer_scales_original(num_layers, layer_decay):
    scales = [0.3 ** (num_layers - i) for i in range(num_layers)]
    scales.append(1.0)
    return scales

# Plotting the two variants:
scales_modified = get_layer_scales(num_layers, layer_decay, decay_type="right_modified", beta=0.4)
scales_original = get_layer_scales_original(num_layers, layer_decay)

plt.figure(figsize=(8,5))
plt.plot(range(num_layers + 1), scales_original, marker='o', label="Original right decay")
plt.plot(range(num_layers + 1), scales_modified, marker='o', label="Modified right (β=0.5)")
plt.xlabel("Layer Index (0 to 24, classification head at 25)")
plt.ylabel("Learning Rate Scale")
plt.title("Comparison of Original and Modified 'Right' Decay")
plt.legend()
plt.grid(True)
plt.show()
#%%

print(scales_modified)

#%%

import matplotlib.pyplot as plt

num_layers = 25
layer_decay = 0.9

# Right Decay: higher rates at later layers.
scales_right = [layer_decay ** (num_layers - i) for i in range(num_layers)]
scales_right.append(1.0)

# Left Decay: higher rates at earlier layers.
scales_left = [layer_decay ** i for i in range(num_layers)]
scales_left.append(1.0)

# Middle Decay: highest rate in the middle, lower toward both ends.
center = num_layers / 2
scales_middle = [layer_decay ** abs(i - center) for i in range(num_layers)]
scales_middle.append(1.0)

# # Left-Middle Decay (version 2): a blend of left and middle biases.
# alpha, beta = 0.5, 0.5
# scales_left_middle = [layer_decay ** (alpha * i + beta * abs(i - center)) for i in range(num_layers)]
# scales_left_middle.append(1.0)

# Plotting: note that we have num_layers + 1 points due to the appended 1.0.
layers = list(range(num_layers + 1))
plt.figure(figsize=(10, 6))
plt.plot(layers, scales_right, marker='o', label="Right Decay")
plt.plot(layers, scales_left, marker='o', label="Left Decay")
plt.plot(layers, scales_middle, marker='o', label="Middle Decay")
#plt.plot(layers, scales_left_middle, marker='o', label="Left-Middle Decay")
plt.xlabel("Layer Index")
plt.ylabel("Learning Rate Scale")
plt.title("Layer-wise Learning Rate Decay Schedules (25 Blocks)")
plt.legend()
plt.grid(True)
plt.show()