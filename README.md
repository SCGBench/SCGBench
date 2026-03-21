# SCGBench: A Service-Oriented Code Generation Benchmark

Welcome to the official repository for **SCGBench**, a diagnostic, two-tiered benchmark designed to evaluate service-oriented code generation. SCGBench bridges the gap between natural-language intent, standardized interface metadata (from RapidAPI), and executable code aligned with real-world GitHub usage patterns.

This repository contains the full dataset of 17,042 real-world service-invocation instances, carefully structured to assess Large Language Models (LLMs) on both fundamental schema reasoning and complex ecosystem integration.

## 📊 Dataset Structure & Files

Based on the diagnostic two-tiered architecture of SCGBench, our data is divided into specific files tailored for different evaluation tasks.

**1. `Template dataset(12416).json` (Tier 1: Canonical Schema Alignment)**
* **Size:** 12,416 instances.
* **Description:** This foundational tier contains highly standardized, template-like invocations.
* **Usage:** Used to evaluate fundamental service-call reasoning and strict schema adherence by isolating the core request from complex business logic.

**2. `Duplicate dataset(4626).json` (Tier 2: Complex Ecosystem Integration)**
* **Size:** 4,626 instances.
* **Description:** This advanced tier consists of deeply contextualized implementations extracted from complex GitHub projects.
* **Usage:** Used to assess an LLM's capacity for real-world code integration (e.g., asynchronous wrappers, cross-file dependencies, UI state management). *(Note: "Duplicate" here refers to the in-the-wild ecosystem implementations replicated from real repositories).*

**3. `Ablation + test experiment dataset(17042).json`**
* **Size:** 17,042 instances.
* **Description:** The complete, merged dataset encompassing both Tier 1 and Tier 2.
* **Usage:** Used for overall baseline evaluations (RQ1) and context ablation studies (RQ2) to test how different levels of contextual information (Docstring, Signature, Parameters, Dependencies) affect code generation.

**4. `Classification experiment dataset(17042).json`**
* **Size:** 17,042 instances.
* **Description:** The complete dataset categorized into 8 distinct service domains (e.g., Gaming & Sports, Infrastructure & DevTools, Content & Media).
* **Usage:** Used to evaluate the impact of service types on LLM performance (RQ3) and diagnose domain-specific pre-training imbalances.

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
JSON
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

<details>
<summary><b>点击此处查看中文说明 (Click here for Chinese translation)</b></summary>

SCGBench: 面向服务的代码生成基准测试
欢迎来到 SCGBench 的官方代码库。这是一个具有诊断性、双层架构的基准测试，旨在评估面向服务的代码生成能力。SCGBench 跨越了自然语言意图、标准化接口元数据（来自 RapidAPI）以及符合 GitHub 真实使用模式的可执行代码之间的鸿沟。

本代码库包含 17,042 个真实世界服务调用实例的完整数据集，经过精心结构化，以评估大型语言模型 (LLM) 在基础模式推理和复杂生态系统集成方面的能力。

📊 数据集结构与文件说明
基于 SCGBench 的诊断性双层架构，我们的数据被划分为适用于不同评估任务的特定文件。

1. Template dataset(12416).json (Tier 1: 规范模式对齐)

规模: 12,416 个实例。

描述: 这一基础层包含高度标准化、类似模板的调用。

用途: 通过将核心请求与复杂的业务逻辑隔离开来，用于评估基础的服务调用推理能力和对严格模式的遵循程度。

2. Duplicate dataset(4626).json (Tier 2: 复杂生态系统集成)

规模: 4,626 个实例。

描述: 这一进阶层由从复杂的 GitHub 项目中提取的、具有深度上下文的实现代码组成。

用途: 用于评估 LLM 在真实代码环境中的集成能力（例如：异步包装器、跨文件依赖、UI 状态管理）。（注：“Duplicate”在此处指代从真实代码库中复制出的具有野生生态系统上下文的实现）。

3. Ablation + test experiment dataset(17042).json

规模: 17,042 个实例。

描述: 包含 Tier 1 和 Tier 2 的完整合并数据集。

用途: 用于整体基准评估 (RQ1) 和上下文消融研究 (RQ2)，以测试不同级别的上下文信息（文档字符串、签名、参数、依赖项）如何影响代码生成。

4. Classification experiment dataset(17042).json

规模: 17,042 个实例。

描述: 按照 8 个不同的服务领域（如游戏与体育、基础设施与开发工具、内容与媒体等）进行分类的完整数据集。

用途: 用于评估服务类型对 LLM 性能的影响 (RQ3)，并诊断特定领域的预训练不平衡问题。

🛠️ 数据构建框架
SCGBench 是通过严谨的多源对齐管道构建的。

API 元数据收集: 使用 Selenium 从 RapidAPI 抓取端点、参数模式和响应定义。

GitHub 源码检索: 从 GitHub 检索实际的 API 调用代码，并使用 AST 解析来定位真实的请求逻辑。

基于 AST 的解耦与验证: 提取目标请求函数，移除无关的冗余上下文，并执行静态语法验证。

LLM 语义补全: 使用 LLM 重建缺失的文档，统一意图表达，并移除个人身份信息 (PII)。

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
SCGBench 中的每个实例都打包为结构化的 JSON 格式。以下是一个标准的数据模板，展示了提取的元数据及对应的代码：
```
JSON
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
