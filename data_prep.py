import os
import json
import shutil
from tqdm import tqdm
lines = []
images = []

for ann in tqdm(os.listdir("./sroie/key")[:20]):
  if ann != ".ipynb_checkpoints":
    with open("./sroie/key/" + ann) as f:
      data = json.load(f)

    images.append(ann[:-4] + "jpg")
    line = {"gt_parse": data}
    lines.append(line)

    with open("./donut/val/metadata.jsonl", 'w') as f:
        for i, gt_parse in enumerate(lines):
            line = {"file_name": images[i], "ground_truth": json.dumps(gt_parse)}
            f.write(json.dumps(line) + "\n")
    shutil.copyfile("./sroie/img/" + images[i], "./donut/val/" + images[i])
