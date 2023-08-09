import csv
import json
from collections import defaultdict

# 定义文件路径
tsv_file_path = "examples/data/test.tsv"  # 替换为你的.tsv文件路径
json_file_path = "examples/data/test.json"  # 转换后的.json文件路径

# 创建一个字典用于存储数据
data_dict = defaultdict(list)

# 打开.tsv文件并读取数据
with open(tsv_file_path, "r", encoding="utf-8") as tsv_file:
    tsv_reader = csv.reader(tsv_file, delimiter="\t")
    next(tsv_reader)  # 跳过首行标题

    for row in tsv_reader:
        if len(row) >= 2:
            instruction = row[0]
            output = row[1]
            if len(output) == 0 or len(instruction) == 0:
                continue
            data_dict[output].append(instruction)

# 限制每个output只保留最多50条数据
for output, instructions in data_dict.items():
    data_dict[output] = instructions[:50]

# 将字典转换为列表
json_data = [{"instruction": inst, "output": out, "input": ""} for out, inst_list in data_dict.items() for inst in
             inst_list]
print("data len:", len(json_data))
# 将数据写入.json文件
with open(json_file_path, "w", encoding="utf-8") as json_file:
    for line in json_data:
        json_file.write(json.dumps(line, ensure_ascii=False) + "\n")
