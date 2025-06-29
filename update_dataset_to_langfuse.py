"""
  程序功能：导入特定的外部数据集到LangFuse特定Project下的dataset中
  @author:charles
"""
import json
import sys
import os
from langfuse import Langfuse
from tqdm import tqdm   #进度条组件

#以下几个环境变量，用于对接langFuse的特定Project
os.environ['LANGFUSE_PUBLIC_KEY']='pk-xxx'
os.environ['LANGFUSE_SECRET_KEY']='sk-lxxx'
os.environ['LANGFUSE_HOST']='<langfuse_host>'

#获得一个本地的数据集
# 这里采用 https://github.com/hiyouga/LLaMA-Factory/blob/main/data/alpaca_zh_demo.json
alpaca_dataset_path = sys.path[1]+"/dataset/alpaca_zh_demo.json"

#创建一个用于上传数据集的集合对象
data_to_upload = []

#读取数据集alpaca_zh_demo.json。注意open的第二个参数模式设置为r,表示Reading
with open(alpaca_dataset_path,'r',encoding='utf-8') as alpaca_ds_file:
    data = alpaca_ds_file.read()
    ds_entries=json.loads(data)
    for entry in ds_entries:
        item = {
            "instruction":entry["instruction"],
            "input":entry["input"],
            "expected_output": entry["output"]
        }
        data_to_upload.append(item)

# init
langfuse = Langfuse()

# 上传到LangFuse
dataset_len=len(data_to_upload)

#langFuse上的数据集名称
ds_name_in_langfuse="Charles测试数据集"

#调用tqdm进度条组件来运行时展示进度
for item in tqdm(data_to_upload[:dataset_len]):
    langfuse.create_dataset_item(
        ## 注意：这个dataset_name需要提前在Langfuse后台创建
        dataset_name=ds_name_in_langfuse,
        input=item["instruction"]+item["input"],
        expected_output=item["expected_output"]
    )
