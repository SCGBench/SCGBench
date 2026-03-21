# SCGBench: A Service-Oriented Code Generation Benchmark

Welcome to the official repository for **SCGBench**, a diagnostic, two-tiered benchmark designed to evaluate service-oriented code generation. SCGBench bridges the gap between natural-language intent, standardized interface metadata (from RapidAPI), and executable code aligned with real-world GitHub usage patterns.

This repository contains the full dataset of 17,042 real-world service-invocation instances, carefully structured to assess Large Language Models (LLMs) on both fundamental schema reasoning and complex ecosystem integration.

## 📊 Dataset Structure & Files

Based on the diagnostic two-tiered architecture of SCGBench, our data is provided in `dataset.zip`, which contains specific files tailored for different evaluation tasks:

**1. `Template dataset(12416).json` (Tier 1: Canonical Schema Alignment)**
* **Size:** 12,416 instances.
* **Usage:** Used to evaluate fundamental service-call reasoning and strict schema adherence by isolating the core request from complex business logic.

**2. `Duplicate dataset(4626).json` (Tier 2: Complex Ecosystem Integration)**
* **Size:** 4,626 instances.
* **Usage:** Used to assess an LLM's capacity for real-world code integration (e.g., asynchronous wrappers, cross-file dependencies).

**3. `Ablation + test experiment dataset(17042).json`**
* **Size:** 17,042 instances.
* **Usage:** Used for overall baseline evaluations (RQ1) and context ablation studies (RQ2).

**4. `Classification experiment dataset(17042).json`**
* **Size:** 17,042 instances.
* **Usage:** Used to evaluate the impact of service types on LLM performance (RQ3).

---

## 🛠️ Data Construction Framework

SCGBench was constructed using a rigorous, multi-source alignment pipeline.

1. **API Metadata Collection:** Scraped endpoints, parameter schemas, and response definitions from RapidAPI using Selenium.
2. **GitHub Source Retrieval:** Retrieved actual API call code from GitHub and used AST parsing to locate real request logic.
3. **AST-Based Decoupling & Validation:** Extracted target request functions, removed irrelevant redundancy, and performed static syntax validation.
4. **LLM Semantic Completion:** Reconstructed missing documentation, unified intent expressions, and removed PII using LLMs.

**Framework Diagram Pipeline:**
```text
[RapidAPI] --(Scrape)--> API Information & Documents
                                |
                                v
[GitHub] --(Search)--> Actual API call code --(AST & Retrieval)--> 17,000+ Request Functions
                                |
                                v
[LLM Processing] --> Generate Comments -> Remove PII -> Format Code -> [SCGBench Dataset]
```
📝 Data Instance Template
Every instance in SCGBench is packaged in a structured JSON format. Below is a standard data template demonstrating the extracted metadata and corresponding code:
```text
{
  "input": "Initializes configuration for fetching...",
  "last_updated": "2025-08-28T14:50:19Z",
  "stars": 10,
  "forks": 5,
  "rely": [
    "requests",
    "json"
  ],
  "function_name": "__init__",
  "parameter": [
    {
      "name": "self",
      "value": null
    }
  ],
  "output": "import requests\nimport json\n\ndef __init__(self):...",
  "language": "python",
  "config_code": "import requests\nimport json",
  "api_type_category": "Content & Media"
}
```
⚙️ Quick Start & Reproduction Guide
Follow these steps to reproduce the evaluation results presented in our paper.

1. Environment Setup
Ensure you have Python 3.8+ installed. Clone this repository and install the required dependencies:
```text
pip install -r requirements.txt
```
2. Data Preparation
Unzip the dataset.zip file located in the root directory to access the JSON datasets required for the experiments.

3. API Key Configuration
Our evaluation scripts use dotenv to load API keys. Create a .env file in the root directory and add your keys for the models you wish to evaluate. For example:
```text
QWEN_API_KEY=your_qwen_key_here
OPENAI_API_KEY=your_openai_key_here
# Add other keys (DeepSeek, Kimi, etc.) as needed
```
4. Running the Experiments
We provide dedicated evaluation scripts for different models (e.g., GPT, Qwen, DeepSeek, Kimi). Before running a script, open the .py file and configure the file paths (e.g., INPUT_JSON, OUTPUT_CSV) to point to your local unzipped datasets.

* **RQ1 (Overall Performance):** Run the Test experiment code([Model]).py scripts using the Ablation + test experiment dataset(17042).json.

* **RQ2 (Ablation Study):** Run the Ablation experiment code([Model]).py scripts using the Ablation + test experiment dataset(17042).json.

* **RQ3 (Service Type Classification):** Run the Classification experiment code([Model]).py scripts using the Classification experiment dataset(17042).json.

Note: The scripts feature auto-saving and checkpointing. If interrupted, simply rerun the script, and it will resume from the last saved state.

<details>
<summary><b>点击此处查看中文说明 (Click here for Chinese translation)</b></summary>

SCGBench: 面向服务的代码生成基准测试
欢迎来到 SCGBench 的官方代码库。这是一个具有诊断性、双层架构的基准测试，旨在评估面向服务的代码生成能力。SCGBench 跨越了自然语言意图、标准化接口元数据（来自 RapidAPI）以及符合 GitHub 真实使用模式的可执行代码之间的鸿沟。

本代码库包含 17,042 个真实世界服务调用实例的完整数据集，经过精心结构化，以评估大型语言模型 (LLM) 在基础模式推理和复杂生态系统集成方面的能力。

📊 数据集结构与文件说明
我们的数据统一打包在 dataset.zip 中，解压后包含适用于不同评估任务的特定文件：

1. Template dataset(12416).json (Tier 1: 规范模式对齐)

* **规模:** 12,416 个实例。

* **用途:** 通过将核心请求与复杂的业务逻辑隔离开来，用于评估基础的服务调用推理能力和对严格模式的遵循程度。

2. Duplicate dataset(4626).json (Tier 2: 复杂生态系统集成)

* **规模:** 4,626 个实例。

* **用途:** 用于评估 LLM 在真实代码环境中的集成能力（例如：异步包装器、跨文件依赖）。

3. Ablation + test experiment dataset(17042).json

* **规模:** 17,042 个实例。

* **用途:** 用于整体基准评估 (RQ1) 和上下文消融研究 (RQ2)。

4. Classification experiment dataset(17042).json

* **规模:** 17,042 个实例。

* **用途:** 按照服务领域分类，用于评估服务类型对 LLM 性能的影响 (RQ3)。

🛠️ 数据构建框架
SCGBench 是通过严谨的多源对齐管道构建的。

1. API 元数据收集: 使用 Selenium 从 RapidAPI 抓取端点、参数模式和响应定义。

2. GitHub 源码检索: 从 GitHub 检索实际的 API 调用代码，并使用 AST 解析来定位真实的请求逻辑。

3. 基于 AST 的解耦与验证: 提取目标请求函数，移除无关的冗余上下文，并执行静态语法验证。

4. LLM 语义补全: 使用 LLM 重建缺失的文档，统一意图表达，并移除个人身份信息 (PII)。

框架流程图:
```text
[RapidAPI] --(抓取)--> API 信息与文档
                                |
                                v
[GitHub] --(搜索)--> 实际 API 调用代码 --(AST 检索)--> 17,000+ 个请求函数
                                |
                                v
[LLM 处理] --> 生成注释 -> 移除 PII -> 格式化代码 -> [SCGBench 数据集]
```
📝 数据实例模板
SCGBench 中的每个实例都打包为结构化的 JSON 格式。以下是一个标准的数据模板：
```text
{
  "input": "Initializes configuration for fetching...",
  "last_updated": "2025-08-28T14:50:19Z",
  "stars": 10,
  "forks": 5,
  "rely": [
    "requests",
    "json"
  ],
  "function_name": "__init__",
  "parameter": [
    {
      "name": "self",
      "value": null
    }
  ],
  "output": "import requests\nimport json\n\ndef __init__(self):...",
  "language": "python",
  "config_code": "import requests\nimport json",
  "api_type_category": "Content & Media"
}
```
⚙️ 快速开始与实验复现指南
请按照以下步骤复现论文中的评估结果。

1. 环境配置
请确保您的环境中安装了 Python 3.8 或以上版本。克隆本仓库后，安装必需的依赖库：
```text
pip install -r requirements.txt
```
2. 数据准备
解压根目录下的 dataset.zip 文件，提取实验所需的 JSON 数据集。

3. API 密钥配置
我们的评估脚本使用 dotenv 来加载 API 密钥。请在项目根目录下创建一个名为 .env 的文件，并填入您要评估的模型的 API 密钥。例如：
```text
QWEN_API_KEY=在此处填入您的qwen密钥
OPENAI_API_KEY=在此处填入您的openai密钥
# 根据需要添加其他模型的密钥 (DeepSeek, Kimi 等)
```
4. 运行实验
我们为不同的模型（如 GPT, Qwen, DeepSeek, Kimi）提供了专门的评估脚本。在运行之前，请打开相应的 .py 文件，修改文件中的路径配置（如 INPUT_JSON, OUTPUT_CSV），将其指向您本地解压后的数据集。

* **RQ1 (整体性能基准):** 使用 Ablation + test experiment dataset(17042).json 数据集运行 Test experiment code([Model]).py 脚本。

* **RQ2 (消融实验):** 使用 Ablation + test experiment dataset(17042).json 数据集运行 Ablation experiment code([Model]).py 脚本。

* **RQ3 (服务类型分类实验):** 使用 Classification experiment dataset(17042).json 数据集运行 Classification experiment code([Model]).py 脚本。

注：脚本内置了自动保存和断点续跑功能。如果运行被中断，重新执行脚本即可从上次保存的进度继续。

