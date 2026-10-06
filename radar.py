import os
import time
import json
import requests
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from openai import OpenAI


# ============================================================
# 基本设置
# ============================================================

SEARCH_DAYS = 7
MAX_RESULTS = 100
TOP_N = 15

PUBMED_QUERY = """
(
    "transitive inference"[Title/Abstract]
    OR "relational inference"[Title/Abstract]
    OR ("hippocampus"[Title/Abstract] AND "prefrontal cortex"[Title/Abstract])
    OR ("ventral hippocampus"[Title/Abstract] AND "prefrontal"[Title/Abstract])
    OR ("hippocampus"[Title/Abstract] AND "anterior cingulate"[Title/Abstract])
    OR ("relational memory"[Title/Abstract] AND "hippocampus"[Title/Abstract])
    OR ("hierarchical representation"[Title/Abstract] AND "hippocampus"[Title/Abstract])
)
"""

HEADERS = {
    "User-Agent": "literature-radar/1.0"
}


# ============================================================
# PubMed 请求
# ============================================================

def pubmed_request(url, params, max_retries=4):

    for attempt in range(max_retries):

        try:

            response = requests.get(
                url,
                params=params,
                headers=HEADERS,
                timeout=30
            )

            if response.status_code == 429:

                wait_time = 5 * (attempt + 1)

                print(
                    f"PubMed rate limit reached. "
                    f"Waiting {wait_time} seconds..."
                )

                time.sleep(wait_time)
                continue

            response.raise_for_status()

            return response

        except requests.RequestException as e:

            if attempt == max_retries - 1:
                raise

            wait_time = 5 * (attempt + 1)

            print(
                f"PubMed request failed: {e}. "
                f"Retrying in {wait_time} seconds..."
            )

            time.sleep(wait_time)

    raise RuntimeError(
        "PubMed request failed after retries."
    )


# ============================================================
# 搜索 PubMed
# ============================================================

def search_pubmed(query, start_date, end_date):

    url = (
        "https://eutils.ncbi.nlm.nih.gov/"
        "entrez/eutils/esearch.fcgi"
    )

    params = {
        "db": "pubmed",
        "term": query,
        "mindate": start_date,
        "maxdate": end_date,
        "datetype": "pdat",
        "retmax": MAX_RESULTS,
        "retmode": "xml",
        "sort": "pub date",
    }

    response = pubmed_request(url, params)

    root = ET.fromstring(response.text)

    return [
        item.text
        for item in root.findall(".//Id")
        if item.text
    ]


# ============================================================
# 获取 PubMed 论文详细信息
# ============================================================

def fetch_pubmed_articles(pmids):

    if not pmids:
        return []

    url = (
        "https://eutils.ncbi.nlm.nih.gov/"
        "entrez/eutils/efetch.fcgi"
    )

    params = {
        "db": "pubmed",
        "id": ",".join(pmids),
        "retmode": "xml",
    }

    response = pubmed_request(url, params)

    root = ET.fromstring(response.text)

    papers = []

    for article in root.findall(".//PubmedArticle"):

        # ----------------------------------------------------
        # Title
        # ----------------------------------------------------

        title_node = article.find(".//ArticleTitle")

        title = ""

        if title_node is not None:
            title = "".join(
                title_node.itertext()
            ).strip()

        # ----------------------------------------------------
        # Abstract
        # ----------------------------------------------------

        abstract_parts = []

        for node in article.findall(
            ".//Abstract/AbstractText"
        ):

            text = "".join(
                node.itertext()
            ).strip()

            if text:

                label = node.attrib.get(
                    "Label"
                )

                if label:
                    abstract_parts.append(
                        f"{label}: {text}"
                    )
                else:
                    abstract_parts.append(
                        text
                    )

        abstract = " ".join(
            abstract_parts
        )

        # ----------------------------------------------------
        # Journal
        # ----------------------------------------------------

        journal = article.findtext(
            ".//Journal/Title",
            default=""
        )

        # ----------------------------------------------------
        # Year
        # ----------------------------------------------------

        year = (
            article.findtext(
                ".//PubDate/Year"
            )
            or article.findtext(
                ".//PubDate/MedlineDate"
            )
            or ""
        )

        # ----------------------------------------------------
        # PMID
        # ----------------------------------------------------

        pmid = article.findtext(
            ".//PMID",
            default=""
        )

        # ----------------------------------------------------
        # DOI
        # ----------------------------------------------------

        doi = ""

        for article_id in article.findall(
            ".//ArticleId"
        ):

            if (
                article_id.attrib.get(
                    "IdType"
                )
                == "doi"
            ):

                doi = (
                    article_id.text
                    or ""
                )

                break

        # ----------------------------------------------------
        # Authors
        # ----------------------------------------------------

        authors = []

        for author in article.findall(
            ".//Author"
        ):

            last_name = author.findtext(
                "LastName",
                default=""
            )

            initials = author.findtext(
                "Initials",
                default=""
            )

            if last_name:

                name = last_name

                if initials:
                    name += (
                        f" {initials}"
                    )

                authors.append(name)

        papers.append({
            "title": title,
            "abstract": abstract,
            "journal": journal,
            "year": year,
            "pmid": pmid,
            "doi": doi,
            "authors": authors,
            "url": (
                "https://pubmed.ncbi.nlm.nih.gov/"
                f"{pmid}/"
            ),
        })

    return papers


# ============================================================
# 去重
# ============================================================

def deduplicate(papers):

    seen = set()
    unique = []

    for paper in papers:

        key = (
            paper["pmid"]
            or paper["doi"]
            or paper["title"].lower()
        )

        if key not in seen:

            seen.add(key)
            unique.append(paper)

    return unique


# ============================================================
# DeepSeek：论文相关性分析
# ============================================================

def analyze_with_deepseek(papers):

    api_key = os.environ.get(
        "OPENAI_API_KEY"
    )

    base_url = os.environ.get(
        "OPENAI_API_BASE",
        "https://api.deepseek.com"
    )

    if not api_key:

        raise RuntimeError(
            "OPENAI_API_KEY is not configured."
        )

    client = OpenAI(
        api_key=api_key,
        base_url=base_url
    )

    # --------------------------------------------------------
    # 为了控制 API 成本，最多分析前 50 篇
    # --------------------------------------------------------

    papers_for_ai = papers[:50]

    paper_text = []

    for i, paper in enumerate(
        papers_for_ai
    ):

        abstract = paper["abstract"]

        if not abstract:
            abstract = "No abstract available."

        paper_text.append(
            f"""
PAPER {i}

Title:
{paper["title"]}

Journal:
{paper["journal"]}

Abstract:
{abstract}
"""
        )

    combined_text = "\n".join(
        paper_text
    )

    system_prompt = """
You are a scientific literature screening assistant.

The researcher is a neuroscience PhD student studying:

1. Transitive inference and relational inference
2. Hippocampal-prefrontal interactions
3. Hippocampus, especially ventral CA1
4. Prelimbic cortex and medial prefrontal cortex
5. Anterior cingulate cortex
6. Relational memory and hierarchical representations
7. Electrophysiology and neural population dynamics
8. Optogenetic circuit manipulation
9. Computational models of value-based learning and relational representation

Your task is to evaluate newly published papers for relevance to this research program.

For every paper:

- Give a relevance score from 0 to 10.
- Identify the main category.
- Explain briefly why it is relevant.
- Give a concise 1-2 sentence TL;DR.

Be conservative.

A paper should receive a high score only if it has a meaningful conceptual, neural-circuit, methodological, behavioral, or computational connection to the research topics above.

Return ONLY valid JSON.

The JSON must be an array with this structure:

[
  {
    "paper_index": 0,
    "score": 8.5,
    "category": "Hippocampus-PFC",
    "why_relevant": "...",
    "tldr": "..."
  }
]
"""

    user_prompt = f"""
Please analyze the following papers.

{combined_text}
"""

    print()
    print("=" * 70)
    print("Sending papers to DeepSeek...")
    print("=" * 70)

    response = client.chat.completions.create(
        model="deepseek-flash",
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ],
        response_format={
            "type": "json_object"
        },
        stream=False
    )

    content = (
        response.choices[0]
        .message
        .content
    )

    result = json.loads(content)

    # DeepSeek 有时可能返回：
    # {"papers": [...]}
    if isinstance(result, dict):

        if "papers" in result:
            result = result["papers"]

        elif "results" in result:
            result = result["results"]

    return result


# ============================================================
# 合并 AI 结果
# ============================================================

def merge_ai_results(
    papers,
    ai_results
):

    analyzed = []

    for result in ai_results:

        index = result.get(
            "paper_index"
        )

        if index is None:
            continue

        if index >= len(papers):
            continue

        paper = papers[index]

        paper["score"] = float(
            result.get(
                "score",
                0
            )
        )

        paper["category"] = result.get(
            "category",
            "Other"
        )

        paper["why_relevant"] = result.get(
            "why_relevant",
            ""
        )

        paper["tldr"] = result.get(
            "tldr",
            ""
        )

        analyzed.append(paper)

    analyzed.sort(
        key=lambda x: x["score"],
        reverse=True
    )

    return analyzed


# ============================================================
# 主程序
# ============================================================

def main():

    end_date = datetime.now(
        timezone.utc
    ).date()

    start_date = (
        end_date
        - timedelta(
            days=SEARCH_DAYS
        )
    )

    start = start_date.strftime(
        "%Y/%m/%d"
    )

    end = end_date.strftime(
        "%Y/%m/%d"
    )

    print("=" * 70)
    print("Weekly Literature Radar")
    print("=" * 70)

    print(
        f"Search period: {start} → {end}"
    )

    # --------------------------------------------------------
    # PubMed
    # --------------------------------------------------------

    print()
    print("Searching PubMed...")

    pmids = search_pubmed(
        PUBMED_QUERY,
        start,
        end
    )

    print(
        f"PubMed found {len(pmids)} papers."
    )

    papers = fetch_pubmed_articles(
        pmids
    )

    papers = deduplicate(
        papers
    )

    print(
        f"Total unique papers: "
        f"{len(papers)}"
    )

    if not papers:

        print(
            "No papers found."
        )

        return

    # --------------------------------------------------------
    # DeepSeek
    # --------------------------------------------------------

    ai_results = analyze_with_deepseek(
        papers
    )

    papers = merge_ai_results(
        papers,
        ai_results
    )

    # --------------------------------------------------------
    # Top papers
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print(
        f"TOP {min(TOP_N, len(papers))} PAPERS"
    )
    print("=" * 70)

    for i, paper in enumerate(
        papers[:TOP_N],
        1
    ):

        print()
        print(
            f"[{i}] "
            f"{paper['score']}/10 "
            f"{paper['title']}"
        )

        print(
            f"Category: "
            f"{paper['category']}"
        )

        print(
            f"Why relevant: "
            f"{paper['why_relevant']}"
        )

        print(
            f"TL;DR: "
            f"{paper['tldr']}"
        )

        print(
            f"URL: "
            f"{paper['url']}"
        )


if __name__ == "__main__":
    main()
