# app.py
import os
import gradio as gr
from openai import OpenAI
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from unsloth import FastLanguageModel
import torch
from ddgs import DDGS 

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
if not DEEPSEEK_API_KEY:
    raise ValueError("请设置环境变量 DEEPSEEK_API_KEY")

client = OpenAI(
    api_key=DEEPSEEK_API_KEY,
    base_url="https://api.deepseek.com"
)

# RAG 向量库
embeddings = HuggingFaceEmbeddings(
    model_name="paraphrase-multilingual-MiniLM-L12-v2",
    cache_folder="./hf_cache"
)
db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)

# 本地微调模型
lora_model = None
lora_tokenizer = None
if os.path.exists("./lora_adapter"):
    try:
        print("正在加载本地微调模型...")
        lora_model, lora_tokenizer = FastLanguageModel.from_pretrained(
            model_name="unsloth/Qwen2.5-1.5B",
            max_seq_length=2048,
            dtype=None,
            load_in_4bit=True,
        )
        lora_model.load_adapter("./lora_adapter")
        print("✅ 本地微调模型加载成功！")
    except Exception as e:
        print(f"⚠️ 加载失败: {e}")
        lora_model = None

# 推理函数
def call_deepseek_api(prompt):
    try:
        resp = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        return resp.choices[0].message.content
    except Exception as e:
        return f"API 错误: {e}"

def call_local_model(prompt):
    if lora_model is None or lora_tokenizer is None:
        return "本地模型未加载"
    inputs = lora_tokenizer(prompt, return_tensors="pt").to("cuda")
    outputs = lora_model.generate(**inputs, max_new_tokens=256)
    return lora_tokenizer.decode(outputs[0], skip_special_tokens=True)

# 联网搜索函数（ddgs）
def web_search(query):
    """使用 DuckDuckGo 搜索，返回前3条结果摘要"""
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(
                query,
                max_results=3,
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            ))
            if not results:
                return "未找到相关结果。"
            snippets = [f"{i+1}. {r['body']}" for i, r in enumerate(results)]
            return "\n".join(snippets)
    except Exception as e:
        return f"搜索错误: {e}"

# Agent 调度
def agent_execute(user_input, model_choice="DeepSeek API"):
    # 计算器
    if any(op in user_input for op in ["+", "-", "*", "/", "乘", "除", "加", "减"]):
        try:
            allowed = set("0123456789+-*/().% ")
            if not all(c in allowed for c in user_input):
                return "错误：包含非法字符"
            return f"计算结果：{eval(user_input)}"
        except Exception as e:
            return f"计算错误：{e}"

    # 检索并获取相似度分数（距离越小越相关）
    docs_with_scores = db.similarity_search_with_score(user_input, k=2)
    RELEVANCE_THRESHOLD = 1.2   # 可根据需要调整（经验值）
    relevant_docs = [(doc, score) for doc, score in docs_with_scores if score < RELEVANCE_THRESHOLD]

    if relevant_docs:
        # 有相关文档，使用 RAG
        context = "\n".join([doc.page_content for doc, _ in relevant_docs])
        prompt = f"请仅根据以下参考资料回答。\n【参考资料】\n{context}\n\n【问题】\n{user_input}"
    else:
        # 本地知识库无相关结果，调用联网搜索
        search_results = web_search(user_input)
        if "未找到相关结果" in search_results or "搜索错误" in search_results:
            return "抱歉，未能在本地知识库和互联网中找到相关信息。请尝试换个问题或检查网络连接。"
        else:
            prompt = f"用户问了一个问题，我进行了联网搜索，得到以下信息：\n{search_results}\n\n请根据这些信息回答用户的问题。如果信息不足，请说明未找到足够资料。"

    # 调用模型
    if model_choice == "本地微调模型" and lora_model is not None:
        return call_local_model(prompt)
    else:
        return call_deepseek_api(prompt)

# Gradio
def chat_fn(message, history, model_choice):
    return agent_execute(message, model_choice)

model_selector = gr.Dropdown(
    choices=["DeepSeek API", "本地微调模型"],
    value="DeepSeek API",
    label="选择模型",
)

demo = gr.ChatInterface(
    fn=chat_fn,
    additional_inputs=[model_selector],
    title="🤖 企业智能助手 (Agent + RAG + 微调 + 联网搜索)",
    description="我可以帮你查内部数据、做算术、联网搜索或闲聊。",
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
