# -*- coding: utf-8 -*-
import os
import time
import random
import requests
from pathlib import Path

from config import DEFAULT_CONFIG

# ===== 初始化配置 =====
BASE_DIR = Path(__file__).resolve().parent.parent
DISHES_DIR = BASE_DIR / "data" / "cook" / "dishes"
API_URL = "https://appbuilder.baidu.com/v2/baike/lemma/get_content"
API_HEADERS = {
    "Content-Type": "application/json",
    "Authorization": DEFAULT_CONFIG.baidu_api_key,
}
DELAY_MIN = 1.5    # 正常请求间隔最小值（秒）
DELAY_MAX = 3.0    # 正常请求间隔最大值（秒）
TIMEOUT = 10
MAX_RETRIES = 3
BACKOFF_BASE = 5    # 429 退避基础等待（秒）
CONSECUTIVE_429_LIMIT = 8  # 连续 429 超过此数则长等待

class BaikeCrawler:

    def extract_dish_names(self, dishes_dir):
        """递归遍历菜品目录，提取所有 .md 文件名（不含后缀）作为菜品名"""
        dish_files = []
        for root, dirs, files in os.walk(dishes_dir):
            for f in files:
                if f.endswith(".md") and not f.startswith("."):
                    filepath = Path(root) / f
                    dish_name = filepath.stem
                    dish_files.append((dish_name, filepath))
        return dish_files


    def already_has_intro(self, filepath, dish_name):
        """检查文件是否已包含百科介绍"""
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                first_line = f.readline().strip()
            return first_line == f"# {dish_name}的介绍"
        except Exception:
            return False


    def search_baike(self, dish_name):
        """通过百度百科 API 搜索菜品词条，返回 summary 文本；失败返回 None"""
        params = {"search_type": "lemmaTitle", "search_key": dish_name}
        last_429 = False

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                resp = requests.get(API_URL, headers=API_HEADERS, params=params, timeout=TIMEOUT)

                if resp.status_code == 429:
                    wait = BACKOFF_BASE * (2 ** (attempt - 1))
                    print(f"    HTTP 429（限流），等待 {wait}s 后重试...")
                    time.sleep(wait)
                    last_429 = True
                    continue

                if resp.status_code != 200:
                    print(f"    HTTP {resp.status_code}")
                    if attempt < MAX_RETRIES:
                        time.sleep(DELAY_MIN * attempt)
                    continue

                data = resp.json()
                result = data.get("result")

                if not result:
                    return None  # 无词条，静默跳过

                summary = result.get("summary", "").strip()
                if summary:
                    return summary
                else:
                    return None

            except requests.exceptions.Timeout:
                print(f"    请求超时，尝试 {attempt}/{MAX_RETRIES}")
                if attempt < MAX_RETRIES:
                    time.sleep(DELAY_MIN * attempt)
            except Exception as e:
                print(f"    异常: {e}，尝试 {attempt}/{MAX_RETRIES}")
                if attempt < MAX_RETRIES:
                    time.sleep(DELAY_MIN * attempt)

        return None


    def update_md_file(self, filepath, dish_name, summary):
        """在 md 文件头部插入百科摘要"""
        intro_header = f"# {dish_name}的介绍"
        intro_block = f"{intro_header}\n{summary}\n\n"

        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()

        new_content = intro_block + content
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(new_content)


    def run(self):
        print("=" * 55)
        print("百度百科菜品爬虫（API 版 v2 — 限流优化）")
        print(f"菜品目录: {DISHES_DIR}")
        print(f"请求间隔: {DELAY_MIN}-{DELAY_MAX}s | 429退避: {BACKOFF_BASE}s 起")
        print("=" * 55)

        dish_files = self.extract_dish_names(DISHES_DIR)
        total = len(dish_files)
        success_count = 0
        skip_count = 0
        consecutive_429 = 0
        long_wait_triggered = 0

        pending = []
        for dish_name, filepath in dish_files:
            if self.already_has_intro(filepath, dish_name):
                success_count += 1  # 计入已成功
            else:
                pending.append((dish_name, filepath))

        print(f"总菜品: {total} | 已完成: {success_count} | 待处理: {len(pending)}")
        print()

        for i, (dish_name, filepath) in enumerate(pending, 1):
            # 连续 429 太多时，长等待恢复
            if consecutive_429 >= CONSECUTIVE_429_LIMIT:
                long_wait = 60
                long_wait_triggered += 1
                print(f"\n⚠️  连续 {consecutive_429} 次 429，等待 {long_wait}s 恢复限额...")
                time.sleep(long_wait)
                consecutive_429 = 0

            print(f"[{i}/{len(pending)}] {dish_name}:", end=" ", flush=True)

            summary = self.search_baike(dish_name)

            if summary:
                self.update_md_file(filepath, dish_name, summary)
                success_count += 1
                print(f"✅")
                consecutive_429 = 0
            else:
                skip_count += 1
                print(f"跳过")

            delay = random.uniform(DELAY_MIN, DELAY_MAX)
            time.sleep(delay)

        print("\n" + "=" * 55)
        print(f"处理完成: 成功 {success_count} 个，跳过 {skip_count} 个")
        if long_wait_triggered:
            print(f"长等待恢复次数: {long_wait_triggered}")
        print("=" * 55)



