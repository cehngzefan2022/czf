import requests
import csv
 
# get SMILES
cids=[]
with open('data/STNN/DrugID_DrugCID_DrugName.csv', 'r', newline='') as csvfile:
    # 创建一个 CSV 读取器
    reader = csv.DictReader(csvfile)
    # 逐行读取数据
    for row in reader:
        # 提取每行中的 CID 并添加到列表中
        cids.append(row['CID'])
 
url = 'https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/property/CanonicalSMILES,title/CSV'
headers = {"content-type":"application/x-www-form-urlencoded"}
data={"cid":cids}
 
res = requests.post(url,data=data,headers=headers)
print (res.text)
 
# write to csv
with open('result.csv', 'w', encoding='UTF8') as f:
    f.write(res.text)
    f.close