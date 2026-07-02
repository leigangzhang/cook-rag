# 项目介绍
这是一个Web爬虫项目，从cook目录下各个菜品的markdown文件提取菜品名，进入百度百科搜索并抓取该菜品的介绍信息，补充到对应的markdown文件中。

## 百度百科API
### API介绍
在「Web网页爬取」方案的基础上，使用「百度百科API」代替Web网页爬取稳定拉取百度百科的菜品词条摘要信息。「百度百科API」的使用示例如下：

```markdown
请求URL：https://appbuilder.baidu.com/v2/baike/lemma/get_content?search_type=lemmaTitle&search_key={菜品名}
请求方式：GET
请求头：{Content-Type: 'application/json', Authorization: 'Bearer <API KEY>'}
返回Response示例：{
    "request_id": "fb65b26e-98a3-417e-a5df-a29012a935ff",
    "result": {
        "lemma_id": 30972,
        "lemma_title": "水煮鱼",
        "lemma_desc": "中国川渝地区的一道特色名菜",
        "url": "https://baike.baidu.com/item/%E6%B0%B4%E7%85%AE%E9%B1%BC/30972?fr=api_ACG_sales",
        "summary": "水煮鱼又称江水煮江鱼、水煮鱼片，是中国川渝地区的一道特色名菜，属于川菜系，其最早流行于重庆市渝北区翠云乡。水煮鱼通常由新鲜草鱼、豆芽、辣椒等食材制作而成。“油而不腻、辣而不燥、麻而不苦、肉质滑嫩”是其特色。",
        "abstract_plain": "水煮鱼又称江水煮江鱼、水煮鱼片，是中国川渝地区的一道特色名菜，属于川菜系，其最早流行于重庆市渝北区翠云乡。\n水煮鱼通常由新鲜草鱼、豆芽、辣椒等食材制作而成。“油而不腻、辣而不燥、麻而不苦、肉质滑嫩”是其特色。\n",
        ...
    }
}
```
菜品摘要的获取方法：在GET请求的URL中替换菜品名，在Response的JSON Body中提取summary。
注意：请求头中的<API KEY>需替换成自己申请的API KEY，申请方式见参考文档部分。

### API优势
- 无反爬拦截（403 问题彻底解决）
- 返回结构化 JSON，提取精准
- 摘要为纯文本，无需清理引用标记
- 响应速度快，延迟可降低到 0.5-1.5s

### 运行效果
运行 `baike_crawler.py` 处理全部 323 个菜品（跳过先前已成功的菜品，断点续传），最终结果：

- ✅ **成功写入 219 个**：获取到百度百科词条摘要，已写入文件头部
- ❌ **跳过 104 个**：部分为非标准菜名在百度百科中无对应词条
- **成功率：67.8%**

### 参考文档
- 百度百科API文档：https://ai.baidu.com/ai-doc/AppBuilder/rmckc6mtu
- 百度百科API KEY申请和管理：https://console.bce.baidu.com/iam/#/iam/apikey/list
- 百度百科API错误码：https://cloud.baidu.com/doc/qianfan-api/s/Om9b4yj3w

