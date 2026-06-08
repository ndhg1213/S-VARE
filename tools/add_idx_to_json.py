import json
import argparse
args = argparse.ArgumentParser()
args.add_argument('--json_file','-j', type=str, default='./inputs/train_church.json')
args.add_argument('--save_path','-s', type=str, default='./inputs/train_church_data')
args = args.parse_args()

# 读取json文件
with open(args.json_file, 'r', encoding='utf-8') as f:
    data = json.load(f)

# 为每个字典加上idx键
for idx, item in enumerate(data):
    item['idx'] = idx
    if 'unsafe_prompt' in item:
        item['original_prompt'] = item['unsafe_prompt']
        del item['unsafe_prompt']
    if 'safe_prompt' in item:
        item['modified_prompt'] = item['safe_prompt']
        del item['safe_prompt']

# 保存回json文件
with open(args.save_path, 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

