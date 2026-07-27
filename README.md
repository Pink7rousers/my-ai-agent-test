# 测试型模型调用试验 (Agent + RAG + LoRA微调 + 联网搜索)

一个基于大语言模型的端到端企业级智能问答系统，集成了检索增强生成 (RAG)、本地 LoRA 微调、工具调用（计算器）和联网搜索，并提供了 Web 界面和 Docker 一键部署。

---

##  主要功能

- **📚 RAG 知识库问答**：基于内部文档（`knowledge.txt`）进行精准回答，有效减少大模型幻觉。
- **🧠 本地微调模型**：使用 LoRA 技术微调 Qwen2.5-1.5B，深度适配客户流失分析等业务场景。
- **🌐 联网搜索**：当内部知识不足时，自动调用 DuckDuckGo 获取最新信息。
- **🔧 工具调用**：支持算术运算等简单工具。
- **🔄 模型切换**：可在 DeepSeek API 和本地微调模型之间自由切换。
- **🐳 容器化部署**：提供 Docker 镜像，一键运行，支持 GPU 加速。


##  快速开始

### 前提条件
- NVIDIA GPU（支持 CUDA 11.8）及 `nvidia-container-toolkit`
- Docker 已安装

### 一键运行（从 Docker Hub 拉取）
```bash
docker run --gpus all -it --rm -p 7860:7860 \
  -e DEEPSEEK_API_KEY="你的DeepSeek密钥" \  
  your-dockerhub-username/my-ai-agent:latest
```
访问 http://localhost:7860 即可使用；当然使用其他api也可以，这里为了方便国内环境使用deepseek。


##  本地构建

```bash
docker build -t my-ai-agent .
docker run --gpus all -it --rm -p 7860:7860 \
  -e DEEPSEEK_API_KEY="你的密钥" \
  my-ai-agent
```


##  技术栈

* Python 3.10 + PyTorch 2.0.1 (CUDA 11.8)
* LangChain + Chroma (RAG)
* Unsloth + PEFT (LoRA)
* Gradio 
* DuckDuckGo 
* Docker 


##  注意事项

* 首次运行会下载嵌入模型（约 400MB），请确保网络畅通。
* 如需使用本地微调模型，请将 lora_adapter 文件夹放入项目目录。


## License

> Copyright (c) 2026 Pink7rousers

This project is licensed under the [MIT License](https://opensource.org/licenses/mit-license.php) - see the [LICENSE](https://github.com/Pink7rousers/my-ai-agent-test/blob/main/LICENSE) file for details.
