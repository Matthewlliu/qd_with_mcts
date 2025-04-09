import os
import json
import shutil

data_dir = "/data1/lhw/qd_mcts/reward_data"
filtered_data_dir = "/data1/lhw/qd_mcts/reward_data_filtered"

correct_samples = 0
total_samples = 0
question_ids = []
for file in os.listdir(data_dir):
    with open(os.path.join(data_dir, file), 'r') as f:
        data = json.load(f)
    total_samples += len(data)
    correct_samples += data[0]["score"]
    if data[0]["score"] > 0:
        question_ids.append(file)
        # copy the file to the filtered_data_dir
        shutil.copy(os.path.join(data_dir, file), os.path.join(filtered_data_dir, file))

print(f"#Question with correct answer: {len(question_ids)}")
print(f"Correct samples: {correct_samples}, Total samples: {total_samples}")
