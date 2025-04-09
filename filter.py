import os
import json
import pandas as pd

filtered_data_dir = "/data1/lhw/qd_mcts/reward_data_filtered"

good_samples = []
bad_samples = []
# extract the "Prompt" column
raw_data = list(pd.read_csv("hf://datasets/google/frames-benchmark/test.tsv", sep="\t")["Prompt"])

for file in os.listdir(filtered_data_dir):
    with open(os.path.join(filtered_data_dir, file), 'r') as f:
        data = json.load(f)
    q_id = int(file.split(".")[0].split("_")[1])
    question = raw_data[q_id]
    for decomposition in data[1:]:
        item = {
            "question": question,
            "decomposition": decomposition["simulation_tree"]
        }
        if decomposition["score"] > 0:
            good_samples.append(item)
        else:
            bad_samples.append(item)

print(f"Good samples: {len(good_samples)}, Bad samples: {len(bad_samples)}")

with open("/data1/lhw/qd_mcts/good_samples.json", "w") as f:
    json.dump(good_samples, f, indent=4)

with open("/data1/lhw/qd_mcts/bad_samples.json", "w") as f:
    json.dump(bad_samples, f, indent=4)
