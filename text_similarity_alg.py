"""
  程序功能：给定2个字符串，要求计算其相似度  （这个在后面实验会用到，比较大模型给出答案以及标准答案的相似度）
  算法：基于文本的余弦相似度
  @author:charles
"""
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

#给定2个文本text_a, text_b,计算余弦相似度
def calc_text_similarity(text_a, text_b):
    #将文本转为向量
    vectorizer = CountVectorizer()
    vectors = vectorizer.fit_transform([text_a,text_b])
    #计算余弦相似度
    similarity=cosine_similarity(vectors)[0][1]
    return similarity

