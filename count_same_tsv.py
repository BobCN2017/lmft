import csv
from collections import defaultdict
from operator import itemgetter

# 定义文件路径
file_path = "examples/backup/train.tsv"  # 将"your_file_path.tsv"替换为你的.tsv文件路径

# 创建一个字典用于统计第二列内容相同的数目
count_dict = defaultdict(int)

# 打开.tsv文件并统计数目
with open(file_path, "r", encoding="utf-8") as tsv_file:
    tsv_reader = csv.reader(tsv_file, delimiter="\t")
    next(tsv_reader)  # 跳过首行标题

    for row in tsv_reader:
        if len(row) >= 2:
            count_dict[row[1]] += 1

# 对字典按数目降序排序
sorted_counts = sorted(count_dict.items(), key=itemgetter(1), reverse=True)

# 打印前100个相同数目最多的内容及数目
print("前100个相同数目最多的内容及数目：")
for idx, (content, count) in enumerate(sorted_counts[:100], 1):
    print(f"{idx}. 内容: '{content}', 数目: {count}")
