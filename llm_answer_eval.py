"""
  程序功能：让大模型做题目，并且和标准答案进行比对，比对的结果通过Langchain回调接口给到Langfuse
  @author:charles
"""
import uuid
from concurrent.futures import ThreadPoolExecutor
from langfuse import Langfuse

from llm_app import llm_application
from text_similarity_alg import calc_text_similarity
import os

#以下几个环境变量，用于对接langFuse的特定Project
os.environ['LANGFUSE_PUBLIC_KEY']='pk-lf-8d07971c-c11a-4a44-8615-acca4cc32b3c'
os.environ['LANGFUSE_SECRET_KEY']='sk-lf-95380ac4-045d-4002-b8e6-32e94fbfb470'
os.environ['LANGFUSE_HOST']='http://127.0.0.1:3000'


langfuse = Langfuse()

#自定义过程：运行llm并评估结果
def run_evaluation(chain, dataset_name, experiment_name):

    #1. 从LangFuse获取指定的数据集
    dataset = langfuse.get_dataset(dataset_name)

    #2.定义一个处理器，用来处理数据集的每一项，处理逻辑是调用calc_text_similarity来计算精度
    def process_item(item):
        #获得处理器名
        handler = item.get_langchain_handler(run_name=experiment_name)
        output = chain.invoke(item.input, config={"callbacks": [handler]})

        #通过调用自定义函数calc_text_similarity来计算出LLM输出和期望输出的差距，并作为精度(accuracy)
        #计算完精度后，会把分数传给langfuse
        langfuse.score(
            name="accuracy",
            value=calc_text_similarity(output, item.expected_output),
            trace_id=handler.get_trace_id(),
        )

    # 通过线程池来并发处理：把内部函数process_item映射到给定数据集的每个项
    with ThreadPoolExecutor(max_workers=4) as executor:
        executor.map(process_item, dataset.items)

#定义在LangFuse上的数据集，比如这里采用测试的数据集"Charles测试数据集"
dataset_name="Charles测试数据集"

#这次实验-数据集测试过程的名称
experiment_name= dataset_name+str(uuid.uuid4())[:8]

#执行评估
run_evaluation(llm_application, dataset_name, experiment_name)