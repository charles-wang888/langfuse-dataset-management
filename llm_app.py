"""
  程序功能：LLM应用，它由三部分组成：基于提示词模板构建的提示词+LLM模型+输出解析器
  @author:charles
"""
from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

#基于提示词模板构建提示词
prompt=PromptTemplate.from_template("""
*********
你是一位Mr Know All先生，世界万物的知识你无所不知。
问个问题:{input}
*********""")

#待测试的模型
model = ChatOpenAI(
        model="qwen-turbo",
        api_key="sk-xxx",
        base_url="<base_url>",
        temperature=0,
        model_kwargs={"seed":42}
    )


#给出输出解析器，这里是StrOutputParser，它会把AIMessage转为str
parser = StrOutputParser()

#创建一个langchain链，它会提示词+模型+解析器  结合起来
llm_application = (
        prompt
        | model
        | parser
)

