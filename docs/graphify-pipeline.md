<!-- markdownlint-disable MD025 -->
# Graphify Pipeline — cook-rag 食谱知识图谱构建流程

## 概述

对 `data/cook/dishes/` 下 **323 道菜谱** 进行程序化结构化提取，构建 **1,550 节点 · 4,677 边** 的知识图谱，完成社区聚类、标注与交互式可视化。

### 运行环境

- Python 3.12 (`/opt/miniconda3/envs/cook-rag-2/bin/python3.12`)
- graphifyy 库

### 为何不用 LLM 提取

原图谱（56 document 节点）的问题在于 `extract_corpus_parallel` 的语义提取管线 `chunk_size` 过大（323 文件仅分 3 个 chunk），导致多数菜谱被降级为 concept 节点。已知菜谱 markdown 格式高度结构化（`#` 标题 → `## 必备原料和工具` → `## 操作`），程序化解析即可完整出图。

---

## Pipeline 步骤

### Step 1 — 检测语料

```bash
python3 -c "
from graphify.detect import detect
from pathlib import Path
result = detect(Path('data/cook/dishes'))
" > graphify-out/.graphify_detect.json
```

结果：323 个 document (.md) + 328 个 image → 只取 document。

### Step 2 — 程序化提取

脚本 [`modules/build_index.py`](/modules/build_index.py)中的`extract_recipe()`方法已经对每个 `.md` 文件执行：

| 抽取项 | 来源 | 示例 |
|--------|------|------|
| 菜名 | `# 标题` 正则去后缀 | `清蒸鲈鱼` |
| 分类 | 父目录名 | `aquatic` |
| 难度 | `烹饪难度：★★★` 星数转 int | 3 |
| 食材 | `## 必备原料和工具` 下 li 项分拆归一化 | `鳜鱼`, `大葱`, `生抽` |
| 技法 | 标题+正文前 800 字中出现的关键词 | `清蒸`, `蒸`, `红烧` |

关键正则：

```python
title_match = re.search(r'^#\s+(.+?)(?:的(?:做法|介绍|制作))?\s*$', content, re.MULTILINE)
diff_match = re.search(r'烹饪难度[：:]\s*(★+|[1-5])', content)
ingr_section = re.search(r'##\s*必备原料和工具\s*\n(.*?)(\n##|\Z)', content, re.DOTALL)
```

产出：323 条 recipe dict，写入 `graphify-out/.graphify_recipes.json`。

### Step 3 — 构建图节点与边

脚本 [`modules/graph_generator.py`](/modules/graph_generator.py) 中的`build_semantic_json()`方法将 recipe 转成 graphify 兼容格式：

**节点类型**（1550 个）：

| 类型 | 数量 | ID 格式 | 示例 |
|------|------|---------|------|
| document | 323 | `rec__<md5_hash>` | `清蒸鲈鱼` |
| concept (分类) | 188 | `cat___<name>` | `aquatic` |
| concept (食材) | 992 | `ingr__<md5_hash>` | `鳜鱼`, `生抽` |
| concept (技法) | 42 | `meth__<name>` | `清蒸`, `红烧` |
| concept (难度) | 5 | `diff__<1-5>` | `中等` |

**边类型**（4677 条）：

| 关系 | 语义 |
|------|------|
| `recipe → category` | `belongs_to` |
| `recipe → difficulty` | `has_difficulty` |
| `recipe → ingredient` | `requires` |
| `recipe → method` | `uses_method` |

去重后写入 `graphify-out/.graphify_extract.json`。

### Step 4 — 聚类 + 社区标注 + 可视化

```python
from graphify.build import build_from_json
from graphify.cluster import cluster, score_all
from graphify.analyze import god_nodes, surprising_connections, suggest_questions
from graphify.report import generate
from graphify.export import to_json, to_html

G = build_from_json(extraction)
communities = cluster(G)       # 111 个社区
cohesion = score_all(G, communities)
gods = god_nodes(G)            # 柱候牛腩 (32 edges) ...
report = generate(...)
to_json(G, communities, 'graphify-out/graph.json')
to_html(G, communities, 'graphify-out/graph.html')
```

**聚类结果**：111 个社区，cohesion 0.045–0.500。

### Step 5 — 收尾

```bash
from graphify.detect import save_manifest
save_manifest(detect['files'])
rm -f graphify-out/.graphify_{detect,ast,analysis,chunk_*}.json
```

---

## 产出文件

| 文件 | 说明 |
|------|------|
| `graphify-out/graph.json` | 完整图谱数据 (1.2 MB) |
| `graphify-out/graph.html` | 交互式可视化 |
| `graphify-out/GRAPH_REPORT.md` | 审计报告 (God Nodes, 意外关联, 建议问题) |
| `graphify-out/manifest.json` | 文件清单 (支持 `--update`) |

---

## 验证：江浙菜完整性

| 菜谱 | 状态 | Edges |
|------|------|-------|
| 清蒸鲈鱼 | ✅ document | ~10 |
| 葱油桂鱼 | ✅ document | ~7 |
| 糖醋鲤鱼 | ✅ document | ~8 |
| 清蒸鳜鱼 | ✅ document | ~9 |
| 醉排骨 | ✅ document | ~8 |
| 葱油拌面 | ✅ document | ~5 |
| 南派红烧肉 | ✅ document | ~7 |
| 炒年糕 | ✅ document | ~6 |
| 醪糟小汤圆 | ✅ document | ~3 |
| 梅菜扣肉 | ✅ document | ~9 |

全部 19 道验证通过，不再只是 concept 引用。

---

## God Nodes (Top 10 by degree)

| 菜谱 | 边数 | 含义 |
|------|------|------|
| 柱候牛腩 | 32 | 柱候酱体系核心 |
| 水煮肉片 | 31 | 川菜技法枢纽 |
| 枝竹羊腩煲 | 30 | 广式煲类代表 |
| 烤鱼 | 28 | 混合调味桥接 |
| 中式馅饼 | 28 | 面食食材富集 |

## 后续建议

1. **启用网络出口** — 对 `api.moonshot.cn` 放行沙箱后，可用 Kimi 做 LLM 级别的语义边（cross-recipe 逻辑关联）
2. **`--update` 增量维护** — 添加新菜谱时运行 `graphify update .` 自动补入
3. **知识查询接入** — 将 `graph.json` 接入 cook-rag 的 `vector_index` 或 Neo4j 做检索增强
