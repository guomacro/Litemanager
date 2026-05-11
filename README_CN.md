# LiteManager - 文献管理智能体

基于 **LangGraph** + **Milvus** + **SQLite** 构建的最小可行文献管理系统。支持 AI 驱动的 PDF 解析、多级摘要、主题图谱组织和语义检索。

## 架构

```
LiteManager/
├── app/
│   ├── config.py              # 统一配置（路径、模型、API Key）
│   ├── state.py               # LangGraph AgentState 状态定义
│   ├── graph.py               # LangGraph 工作流编排
│   ├── llm_service.py         # LLM 调用（分类、摘要、主题路径、查询改写）
│   ├── nodes/
│   │   ├── route_intent.py    # 意图路由节点
│   │   ├── parse_pdf.py       # PDF 文本/元数据提取
│   │   ├── summarize.py       # 文献类型识别 + 结构化摘要
│   │   ├── build_graph.py     # 构建主题/类型图关系
│   │   ├── index_milvus.py    # 文本切块 + Embedding + 写入 Milvus
│   │   ├── persist_metadata.py # 保存元数据 + 摘要到 SQLite
│   │   ├── delete_paper.py    # 级联删除（向量、图、元数据、文件）
│   │   └── search_papers.py   # 查询改写 + 摘要优先搜索 + 答案生成
│   └── services/
│       ├── pdf_service.py     # PyMuPDF PDF 解析服务
│       ├── milvus_service.py  # Milvus 向量存储服务（chunks + summaries 集合）
│       ├── metadata_service.py # SQLite 元数据 + 图谱 CRUD
│       └── graph_service.py   # 论文-主题-概念图谱构建
├── gui/
│   └── app.py                 # Gradio Web 界面（5 个标签页）
├── prompts/
│   ├── summarize.md           # 摘要 prompt 模板
│   └── classify.md            # 分类 prompt 模板
├── data/                      # 运行时数据（SQLite、Milvus、PDF 副本）
├── requirements.txt
└── README.md
```

## LangGraph 工作流

```
                    ┌─────────────┐
                    │ 路由意图判断 │
                    └──────┬──────┘
           ┌───────────────┼───────────────┬──────────────┐
           ▼               ▼               ▼              ▼
    ┌──────────┐    ┌──────────┐    ┌──────────┐   ┌──────────┐
    │ 导入流程  │    │ 删除流程  │    │ 检索流程  │   │ 总结流程  │
    └────┬─────┘    └────┬─────┘    └────┬─────┘   └────┬─────┘
         │               │               │              │
    解析PDF          定位论文          查询改写        类型识别
         │               │               │              │
    类型识别            删除向量        摘要检索        生成摘要
         │               │               │              │
    生成摘要            删除图数据      分块检索        持久化元数据
         │               │               │              │
    持久化元数据        删除元数据      生成答案           END
         │               │               │
    构建图谱            删除文件          END
         │               │
    向量入库             END
         │
         END
```

### 导入流程
1. `parse_pdf` - 通过 PyMuPDF 提取文本、标题、摘要、关键词、章节
2. `detect_doc_type` - LLM 分类文献类型（survey/method/application/benchmark/theory）并生成 topic_path
3. `summarize` - 生成结构化摘要（paper_summary、key_points、section_summaries）
4. `persist_metadata` - 论文记录 + 摘要写入 SQLite，PDF 复制到 data/
5. `build_graph` - 创建图节点（论文、主题层级、文献类型）和边关系
6. `index_milvus` - 文本切块，Embedding，写入 Milvus 的 `paper_chunks` 和 `paper_summaries` 集合

### 检索流程
1. `search_papers` - LLM 改写查询，先搜摘要再搜分块，去重排序
2. `generate_answer` - LLM 综合结果生成简洁回答

### 删除流程
1. `delete_paper` - 定位论文；软删除（标记已删除）或硬删除（级联清理向量、图、文件、元数据）

## 数据存储

| 层级 | 技术 | 用途 |
|------|------|------|
| 元数据 | SQLite | 论文记录、图节点/边、摘要 |
| 向量 | Milvus | `paper_chunks` 集合（文本块 + 向量）和 `paper_summaries` 集合 |
| 文件 | 本地磁盘 | PDF 文件存储在 `data/papers/` |

### SQLite 表结构
- **papers**: paper_id, title, authors, abstract, keywords, source, doc_type, topic_path, file_path, status, timestamps
- **paper_summaries**: paper_id, summary_level, summary_data (JSON), created_at
- **graph_nodes**: node_id, node_type (paper/topic/doc_type), label, props
- **graph_edges**: edge_id, source_id, target_id, edge_type (belongs_to/subtopic_of/has_type), props

### Milvus 集合
- **paper_chunks**: chunk_id, paper_id, chunk_index, chunk_text, embedding (1536d), topic_path, doc_type
- **paper_summaries**: summary_id, paper_id, summary_level, summary_text, embedding (1536d), topic_path

## 环境配置

```bash
conda create -n agent3.12 python=3.12
conda activate agent3.12
pip install -r requirements.txt
```

配置 OpenAI 兼容 API：
```bash
export OPENAI_API_KEY="your-api-key"
export OPENAI_BASE_URL="https://api.openai.com/v1"
export LLM_MODEL="gpt-4o-mini"
export EMBED_MODEL="text-embedding-3-small"
```

Milvus 默认使用 **Lite** 模式（嵌入式，无需服务端）。生产环境可切换：
```bash
export MILVUS_USE_LITE=false
export MILVUS_HOST=localhost
export MILVUS_PORT=19530
```

## 使用方式

```bash
python -m gui.app
```

浏览器打开 http://127.0.0.1:7860

### GUI 功能标签
| 标签 | 功能 |
|------|------|
| Import PDF | 上传 PDF，选择摘要级别（quick/standard/deep），触发完整导入流程 |
| Search | 语义搜索，支持主题/类型过滤，AI 综合回答 |
| Library | 浏览全部论文，支持关键词过滤 |
| View Details | 查看论文完整元数据、摘要和图关系 |
| Delete | 软删除（可恢复）或硬删除（永久级联清理） |

## MVP 功能清单
- [x] PDF 导入，自动提取元数据（标题、摘要、关键词、章节）
- [x] AI 文献类型分类（survey/method/application/benchmark/theory）
- [x] 自动 topic_path 生成（如 `AI/NLP/Question-Answering/RAG`）
- [x] 多级结构化摘要（quick/standard/deep）
- [x] 基于图的话题组织（论文 -> 主题 -> 子主题层级）
- [x] Milvus 向量索引（chunks + summaries，余弦相似度）
- [x] 语义搜索，查询改写 + AI 综合回答
- [x] 软/硬删除，级联清理（向量、图、元数据、文件）
- [x] Gradio Web 界面

## 路线图
1. ✅ **第一阶段（MVP）**: 导入、解析、摘要、检索、删除
2. **第二阶段（Graph RAG）**: 增强主题图遍历 + 向量混合检索
3. **第三阶段（生命周期）**: 批量操作、导出、版本历史
4. **第四阶段（外部搜索）**: Arxiv API 集成、候选论文导入
5. **第五阶段（增强）**: OCR 支持、Neo4j 迁移、多用户
