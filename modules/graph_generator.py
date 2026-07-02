#!/usr/bin/env python3
"""读取含有元数据的菜谱列表 → 建图 → 聚类 → 社区标注 → 报告 → HTML。

前置条件：graphify_out/.graphify_extract.json 已由 extract_recipes.py 生成。
"""

import hashlib
import json
import logging
from pathlib import Path

from graphify.build import build_from_json
from graphify.cluster import cluster, score_all
from graphify.analyze import god_nodes, surprising_connections, suggest_questions
from graphify.report import generate
from graphify.export import to_json, to_html
from sympy import true


GRAPHIFY_DIR = Path("graphify-out")
RECIPES_FILE = GRAPHIFY_DIR / ".graphify_recipes.json"
EXTRACT_FILE = GRAPHIFY_DIR / ".graphify_extract.json"
DETECT_FILE = GRAPHIFY_DIR / ".graphify_detect.json"
GRAPH_FILE = GRAPHIFY_DIR / "graph.json"
HTML_FILE = GRAPHIFY_DIR / "graph.html"
REPORT_FILE = GRAPHIFY_DIR / "GRAPH_REPORT.md"

logger = logging.getLogger(__name__)


def build_semantic_json(recipes: list[dict]) -> None:
    """将 recipe 列表转成 graphify 兼容的 .graphify_extract.json 结构。


    包含的节点类型：
      - document: 每道菜一个节点
      - concept (category): 菜品分类
      - concept (difficulty): 难度等级
      - concept (cuisine): 菜系
      - concept (time): 烹饪时间分桶
      - concept (carb): 碳水含量分桶
      - concept (ingredient): 食材
      - concept (method): 烹饪技法

    边关系：
      - belongs_to:          recipe → category
      - has_difficulty:       recipe → difficulty
      - belongs_to_cuisine:  recipe → cuisine
      - cooking_time:        recipe → time bucket
      - carb_level:          recipe → carb bucket
      - requires:            recipe → ingredient
      - uses_method:         recipe → method
    """

    nodes: list[dict] = []
    edges: list[dict] = []

    # ---- 分类节点 ----------------------------------------------------
    categories = sorted({r["category"] for r in recipes})
    for cat in categories:
        nodes.append({
            "id": f"cat___{cat}",
            "label": cat,
            "file_type": "concept",
            "source_file": None,
            "source_location": None,
        })

    # ---- 难度节点 ----------------------------------------------------
    difficulty_labels = {
        1: "很简单", 2: "简单", 3: "中等", 4: "较难", 5: "困难",
    }
    for d in range(1, 6):
        nodes.append({
            "id": f"diff__{d}",
            "label": difficulty_labels[d],
            "file_type": "concept",
            "source_file": None,
            "source_location": None,
        })

    # ---- 菜系节点 --------------------------------------------------
    cuisine_set: set[str] = set()
    for r in recipes:
        for c in r.get("cuisines", []):
            cuisine_set.add(c)
    for cuisine in sorted(cuisine_set):
        nodes.append({
            "id": f"cuisine___{cuisine}",
            "label": cuisine,
            "file_type": "concept",
            "source_file": None,
            "source_location": None,
        })

    # ---- 烹饪时间节点 ------------------------------------------------
    time_buckets: dict[str, str] = {
        "快速(<15分钟)": "<15min",
        "中速(15-30分钟)": "15-30min",
        "慢速(>30分钟)": ">30min",
    }
    for label, bid in time_buckets.items():
        nodes.append({
            "id": f"time___{bid}",
            "label": label,
            "file_type": "concept",
            "source_file": None,
            "source_location": None,
        })

    # ---- 碳水含量节点 ------------------------------------------------
    carb_buckets: dict[str, str] = {
        "无糖(<5g)": "<5g",
        "低碳(5-20g)": "5-20g",
        "正常(>20g)": ">20g",
    }
    for label, bid in carb_buckets.items():
        nodes.append({
            "id": f"carb__{bid}",
            "label": label,
            "file_type": "concept",
            "source_file": None,
            "source_location": None,
        })

    # ---- 食材 & 技法概念节点 -----------------------------------------
    ingredient_set: set[str] = set()
    method_set: set[str] = set()
    for r in recipes:
        for ing in r["ingredients"]:
            ingredient_set.add(ing)
        for mtd in r["methods"]:
            method_set.add(mtd)

    for ing in sorted(ingredient_set):
        node_id = "ingr__" + hashlib.md5(ing.encode()).hexdigest()[:12]
        nodes.append({
            "id": node_id,
            "label": ing,
            "file_type": "concept",
            "source_file": None,
            "source_location": None,
        })

    for mtd in sorted(method_set):
        nodes.append({
            "id": f"meth__{mtd}",
            "label": mtd,
            "file_type": "concept",
            "source_file": None,
            "source_location": None,
        })

    # ---- 菜谱 document 节点 + 边 -------------------------------------
    for r in recipes:
        recipe_id = r.get("recipe_id")
        nodes.append({
            "id": recipe_id,
            "label": r["dish_name"],
            "file_type": "document",
            "source_file": r["source_file"],
            "source_location": None,
        })

        edges.append({
            "source": recipe_id,
            "target": f'cat___{r["category"]}',
            "type": "belongs_to",
            "label": "belongs_to",
        })

        edges.append({
            "source": recipe_id,
            "target": f'diff__{r["difficulty"]}',
            "type": "has_difficulty",
            "label": "has_difficulty",
        })

        for ing in r["ingredients"]:
            ing_id = "ingr__" + hashlib.md5(ing.encode()).hexdigest()[:12]
            edges.append({
                "source": recipe_id,
                "target": ing_id,
                "type": "requires",
                "label": "requires",
            })

        for mtd in r["methods"]:
            edges.append({
                "source": recipe_id,
                "target": f"meth__{mtd}",
                "type": "uses_method",
                "label": "uses_method",
            })

        for cuisine in r.get("cuisines", []):
            edges.append({
                "source": recipe_id,
                "target": f"cuisine___{cuisine}",
                "type": "belongs_to_cuisine",
                "label": "belongs_to_cuisine",
            })

        ct = r.get("cooking_time_min")
        if ct is not None:
            if ct < 15:
                bid = "<15min"
            elif ct <= 30:
                bid = "15-30min"
            else:
                bid = ">30min"
            edges.append({
                "source": recipe_id,
                "target": f"time___{bid}",
                "type": "cooking_time",
                "label": "cooking_time",
            })

        carb = r.get("carb_amount_g")
        if carb is not None:
            if carb < 5:
                bid = "<5g"
            elif carb <= 20:
                bid = "5-20g"
            else:
                bid = ">20g"
            edges.append({
                "source": recipe_id,
                "target": f"carb__{bid}",
                "type": "carb_level",
                "label": "carb_level",
            })

    # ---- 去重边 ------------------------------------------------------
    seen: set[tuple] = set()
    unique_edges: list[dict] = []
    for e in edges:
        key = (e["source"], e["target"], e["type"])
        if key not in seen:
            seen.add(key)
            unique_edges.append(e)

    result = {
        "nodes": nodes,
        "edges": unique_edges,
        "hyperedges": [],
        "input_tokens": 0,
        "output_tokens": 0,
    }

    EXTRACT_FILE.parent.mkdir(parents=True, exist_ok=True)
    EXTRACT_FILE.write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8",
    )
    logger.info(f"\nWritten to {EXTRACT_FILE}")



def label_communities_auto(G, communities):
    """基于每个社区内度数最高的节点自动命名。"""
    labels = {}
    for cid, node_ids in sorted(communities.items()):
        top = sorted(
            [(nid, G.degree(nid), G.nodes.get(nid, {}).get("label", nid))
             for nid in node_ids],
            key=lambda x: (-x[1], x[2]),
        )
        docs = [label for _, _, label in top if not label.startswith("Community")]
        labels[cid] = docs[0] if docs else f"群组{cid}"
    return labels


def run():
    # ---- 将recipe列表 转成 graphify 兼容Json格式 ---------------------------------------------------------
    logger.info("Build Semantic ...")
    recipes = json.loads(RECIPES_FILE.read_text(encoding="utf-8"))
    build_semantic_json(recipes)
    
    logger.info("Loading extraction ...")
    extraction = json.loads(EXTRACT_FILE.read_text(encoding="utf-8"))

    # 如果检测文件存在则读取，否则构造最小结构
    if DETECT_FILE.exists():
        detection = json.loads(DETECT_FILE.read_text(encoding="utf-8"))
    else:
        detection = {
            "files": {"document": []},
            "total_files": extraction.get("total_files", 0),
            "total_words": extraction.get("total_words", 0),
            "skipped_sensitive": [],
            "needs_graph": True,
        }

    tokens = {
        "input": extraction.get("input_tokens", 0),
        "output": extraction.get("output_tokens", 0),
    }

    # ---- 建图 ---------------------------------------------------------
    logger.info("Building graph ...")
    G = build_from_json(extraction)
    logger.info(f"  Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # ---- 聚类 ---------------------------------------------------------
    logger.info("Clustering ...")
    communities = cluster(G)
    cohesion = score_all(G, communities)
    logger.info(f"  Communities: {len(communities)}")

    # ---- 分析 ---------------------------------------------------------
    logger.info("Analyzing ...")
    gods = god_nodes(G)
    surprises = surprising_connections(G, communities)
    labels = label_communities_auto(G, communities)
    questions = suggest_questions(G, communities, labels)

    logger.info(f"  Top God Nodes:")
    for gn in gods[:10]:
        logger.info(f"    {gn['label']} ({gn['degree']} edges)")

    # ---- 报告 ---------------------------------------------------------
    logger.info("Generating report ...")
    report = generate(
        G, communities, cohesion, labels, gods, surprises,
        detection, tokens, ".",
        suggested_questions=questions,
    )
    REPORT_FILE.write_text(report, encoding="utf-8")
    logger.info(f"  → {REPORT_FILE}")

    # ---- 导出 JSON + HTML ---------------------------------------------
    logger.info("Exporting ...")
    to_json(G, communities, str(GRAPH_FILE), force=True)
    logger.info(f"  → {GRAPH_FILE}")
    to_html(G, communities, str(HTML_FILE))
    logger.info(f"  → {HTML_FILE}")

    # ---- 汇总 ---------------------------------------------------------
    docs = sum(
        1 for _, d in G.nodes(data=True)
        if d.get("file_type") == "document"
    )
    concepts = G.number_of_nodes() - docs
    logger.info(f"\nDone.")
    logger.info(f"  Documents:  {docs}")
    logger.info(f"  Concepts:   {concepts}")
    logger.info(f"  Total:      {G.number_of_nodes()} nodes")
    logger.info(f"  Edges:      {G.number_of_edges()}")
    logger.info(f"  Communities:{len(communities)}")


if __name__ == "__main__":
    run()
