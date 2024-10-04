import itertools
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch

#######################S2
'''datafile = 'drugbank'
positive_combinations = pd.read_csv('data/' + datafile + '.csv')
# positive_combinations = df_combinations[df_combinations['label'] == 1]

# 获取所有唯一的药物标识符
df_drug_list = pd.read_csv('data/DDI_613_SMILE.csv')#612
all_drugs = df_drug_list['drugID'].tolist()
# all_drugs = pd.unique(positive_combinations[['drug1', 'drug2']].values.ravel('K'))#596
print(len(all_drugs))
# 将药物标识符按照4:1的比例划分成两份
train_drugs, val_drugs = train_test_split(all_drugs, test_size=0.2)
# 创建包含药物的数据框
train_df = pd.DataFrame({'Drug': train_drugs})
val_df = pd.DataFrame({'Drug': val_drugs})

# 保存为CSV文件
train_df.to_csv('data/inductive/S1/train_drugs.csv', index=False)
val_df.to_csv('data/inductive/S1/val_drugs.csv', index=False)

# 使用划分好的药物标识符获取对应的训练集和验证集
train_set = positive_combinations[(positive_combinations['drug1'].isin(train_drugs)) & (positive_combinations['drug2'].isin(train_drugs))]
test_set = positive_combinations[(positive_combinations['drug1'].isin(val_drugs)) & (positive_combinations['drug2'].isin(val_drugs))]

# 输出训练集和验证集的样本数量
print(f"训练集样本数量: {len(train_set)}")
print(f"验证集样本数量: {len(test_set)}")

# 保存训练集和验证集为 CSV 文件
train_set.to_csv('data/inductive/S1/drugbank_train_inductive.csv', index=False)
test_set.to_csv('data/inductive/S1drugbank_test_inductive.csv', index=False)


'''


#######################S1
datafile = 'drugbank'
positive_combinations = pd.read_csv('data/' + datafile + '.csv')
# positive_combinations = df_combinations[df_combinations['label'] == 1]

# 获取所有唯一的药物标识符
df_drug_list = pd.read_csv('data/DDI_613_SMILE.csv')#612
all_drugs = df_drug_list['drugID'].tolist()
# all_drugs = pd.unique(positive_combinations[['drug1', 'drug2']].values.ravel('K'))#596
print(len(all_drugs))
# 将药物标识符按照4:1的比例划分成两份
train_drugs, val_drugs = train_test_split(all_drugs, test_size=0.1)
# 创建包含药物的数据框
train_df = pd.DataFrame({'Drug': train_drugs})
val_df = pd.DataFrame({'Drug': val_drugs})

# 保存为CSV文件
train_df.to_csv('data/inductive/S1/train_drugs.csv', index=False)
val_df.to_csv('data/inductive/S1/test_drugs.csv', index=False)

# 使用划分好的药物标识符获取对应的训练集和验证集
train_combinations, val_combinations = train_test_split(positive_combinations, test_size=0.2, random_state=42)

#确保每个训练集组合的两种药物都不出现在验证集中
train_set = train_combinations[~train_combinations.apply(lambda row: ((row['drug1'] in val_drugs) or (row['drug2'] in val_drugs)), axis=1)]

# 确保每个验证集组合的两种药物至少有一种不出现在训练集中
val_set = val_combinations[~val_combinations.apply(lambda row: ((row['drug1'] in train_drugs) and (row['drug2'] in train_drugs)), axis=1)]

# 输出训练集和验证集的样本数量
print(f"训练集样本数量: {len(train_set)}")
print(f"验证集样本数量: {len(val_set)}")

# 保存训练集和验证集为 CSV 文件
train_set.to_csv('data/inductive/S1/drugbank_train_inductive.csv', index=False)
val_set.to_csv('data/inductive/S1/drugbank_test_inductive.csv', index=False)


