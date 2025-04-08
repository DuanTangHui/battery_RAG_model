import os
os.environ["STREAMLIT_WATCHER_TYPE"] = "none"
import streamlit as st
import nest_asyncio
import asyncio
from langchain_openai import ChatOpenAI
import os
from m3e import M3EBaseEmbeddings
from langchain_chroma import Chroma
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

# 应用 nest_asyncio 以解决事件循环问题
nest_asyncio.apply()

# 确保事件循环正确初始化
loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)

def get_vectordb():
    embedding = M3EBaseEmbeddings(model_name="moka-ai/m3e-base")
    persist_directory = "./data_base/vector_db/chroma"
    vectordb = Chroma(
        persist_directory=persist_directory,
        embedding_function=embedding
    )
    return vectordb

def get_chat_qa_chain(question: str):
    vectordb = get_vectordb()
    deepseek_api_base = "https://api.deepseek.com/v1"
    llm = ChatOpenAI(
        model_name="deepseek-chat",
        openai_api_base=deepseek_api_base,
    )
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
    )
    retriever = vectordb.as_retriever()
    qa = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
    )
    result = qa.invoke({"question": question})
    return result["answer"]

def get_qa_chain(question: str):
    vectordb = get_vectordb()
    deepseek_api_base = "https://api.deepseek.com/v1"
    llm = ChatOpenAI(
        model_name="deepseek-chat",
        openai_api_base=deepseek_api_base,
    )
    template = """
    使用以下上下文来回答最后的问题。如果你不知道答案，就说你不知道，不要试图编造答案。
    最多使用三句话。尽量使答案简明扼要。总是在回答的最后说“谢谢你的提问！”。
    {context}
    问题: {question}
    """
    QA_CHAIN_PROMPT = PromptTemplate(
        input_variables=["context", "question"],
        template=template
    )
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectordb.as_retriever(),
        return_source_documents=True,
        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT},
    )
    result = qa_chain.invoke({"query": question})
    return result["result"]

def main():
    st.title('🦜🔗 电池领域的RAG模型')
    deepseek_api = st.sidebar.text_input('Deepseek API Key', type='password')
    with st.form('my_form'):
        selected_method = st.radio(
            "你想选择哪种模式进行对话？",
            ["None", "qa_chain", "chat_qa_chain"],
            captions=["不使用检索问答的普通模式", "不带历史记录的检索问答模式", "带历史记录的检索问答模式"]
        )
        text = st.text_area('Enter text:', '电池可以研究哪些方面？')
        submitted = st.form_submit_button('Submit')
        if not deepseek_api.startswith('sk-'):
            st.warning('Please enter your OpenAI API key!', icon='⚠')
        if submitted and deepseek_api.startswith('sk-'):
            os.environ["OPENAI_API_KEY"] = deepseek_api
            if selected_method == "None":
                st.info(get_qa_chain(text))
            elif selected_method == "qa_chain":
                st.info(get_qa_chain(text))
            else:
                st.info(get_chat_qa_chain(text))

if __name__ == "__main__":
    main()