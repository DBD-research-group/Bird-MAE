#%%
import pandas as pd 
from collections import Counter
import random

from datasets import load_dataset, Audio, DatasetDict
from birdset.datamodule.components.event_mapping import XCEventMapping

from tqdm import tqdm 

def smart_sampling( dataset, label_name, class_limit, event_limit):
    def _unique_identifier(x, labelname):
        file = x["filepath"]
        label = x[labelname]
        return {"id": f"{file}-{label}"}

    class_limit = class_limit if class_limit else -float("inf")
    dataset = dataset.map(
        lambda x: _unique_identifier(x, label_name), desc="smart-sampling-unique-id"
    )
    df = pd.DataFrame(dataset)
    path_label_count = df.groupby(["id", label_name], as_index=False).size()
    path_label_count = path_label_count.set_index("id")
    class_sizes = df.groupby(label_name).size()

    for label in tqdm(class_sizes.index, desc="smart-sampling"):
        current = path_label_count[path_label_count[label_name] == label]
        total = current["size"].sum()
        most = current["size"].max()

        while total > class_limit or most != event_limit:
            largest_count = current["size"].value_counts()[current["size"].max()]
            n_largest = current.nlargest(largest_count + 1, "size")
            to_del = n_largest["size"].max() - n_largest["size"].min()

            idxs = n_largest[n_largest["size"] == n_largest["size"].max()].index
            if (
                total - (to_del * largest_count) < class_limit
                or most == event_limit
                or most == 1
            ):
                break
            for idx in idxs:
                current.at[idx, "size"] = current.at[idx, "size"] - to_del
                path_label_count.at[idx, "size"] = (
                    path_label_count.at[idx, "size"] - to_del
                )

            total = current["size"].sum()
            most = current["size"].max()

    event_counts = Counter(dataset["id"])

    all_file_indices = {label: [] for label in event_counts.keys()}
    for idx, label in enumerate(dataset["id"]):
        all_file_indices[label].append(idx)

    limited_indices = []
    for file, indices in all_file_indices.items():
        limit = path_label_count.loc[file]["size"]
        limited_indices.extend(random.sample(indices, limit))

    dataset = dataset.remove_columns("id")
    return dataset.select(limited_indices)


ds = load_dataset("DBD-research-group/BirdSet", "XCM", cache_dir="/home/anonymous/projects/birdMAE/data/XCM", num_proc=4)
ds["train"] = ds["train"].select(range(10_000))

ds = ds["train"].cast_column(
    column="audio",
    feature=Audio(
        sampling_rate=32_000,
        mono=True,
        decode=False,
    ),
)

mapper = XCEventMapping()

ds = ds.map(
    mapper,
    remove_columns=["audio"],
    batched=True,
    batch_size=300,
    num_proc=3,
    desc="Train event mapping"
)


ds = smart_sampling(
    dataset=ds,
    label_name="ebird_code",
    class_limit=100,
    event_limit=1
)

ds = ds.select_columns(["filepath", "ebird_code_multilabel", "detected_events", "start_time", "end_time"])

ds_dict = DatasetDict({
    "train": ds,
})


ds_dict.save_to_disk("/home/anonymous/projects/birdMAE/data/XCM/XCM_processed_100_1events_ogg")
