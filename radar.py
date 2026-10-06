import requests
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone


# =========================
# 基本设置
# =========================

SEARCH_DAYS = 7
MAX_RESULTS = 30

SEARCH_TERMS = [
    '"transitive inference"[Title/Abstract]',
    '"relational inference"[Title/Abstract]',
    '("hippocampus"[Title/Abstract] AND "prefrontal cortex"[Title/Abstract])',
    '("ventral hippocampus"[Title/Abstract] AND "prefrontal"[Title/Abstract])',
    '("hippocampus"[Title/Abstract] AND "anterior cingulate"[Title/Abstract])',
    '("relational memory"[Title/Abstract] AND "hippocampus"[Title/Abstract])',
    '("hierarchical representation"[Title/Abstract] AND "hippocampus"[Title/Abstract])',
]


# =========================
# PubMed 搜索
# =========================

def search_pubmed(query, start_date, end_date):

    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"

    params = {
        "db": "pubmed",
        "term": query,
        "mindate": start_date,
        "maxdate": end_date,
        "datetype": "pdat",
        "retmax": MAX_RESULTS,
        "retmode": "xml",
    }

    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()

    root = ET.fromstring(response.text)

    return [
        item.text
        for item in root.findall(".//Id")
        if item.text
    ]


# =========================
# 获取论文详细信息
# =========================

def fetch_pubmed_articles(pmids):

    if not pmids:
        return []

    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

    params = {
        "db": "pubmed",
        "id": ",".join(pmids),
        "retmode": "xml",
    }

    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()

    root = ET.fromstring(response.text)

    papers = []

    for article in root.findall(".//PubmedArticle"):

        title = article.findtext(".//ArticleTitle", default="")

        abstract_parts = []

        for node in article.findall(".//Abstract/AbstractText"):
            if node.text:
                abstract_parts.append(node.text)

        abstract = " ".join(abstract_parts)

        journal = article.findtext(
            ".//Journal/Title",
            default=""
        )

        year = (
            article.findtext(".//PubDate/Year")
            or article.findtext(".//PubDate/MedlineDate")
            or ""
        )

        pmid = article.findtext(".//PMID", default="")

        doi = ""

        for article_id in article.findall(".//ArticleId"):
            if article_id.attrib.get("IdType") == "doi":
                doi = article_id.text or ""

        papers.append({
            "title": title,
            "abstract": abstract,
            "journal": journal,
            "year": year,
            "pmid": pmid,
            "doi": doi,
            "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
        })

    return papers


# =========================
# 去重
# =========================

def deduplicate(papers):

    seen = set()
    unique = []

    for paper in papers:

        key = paper["pmid"]

        if key not in seen:
            seen.add(key)
            unique.append(paper)

    return unique


# =========================
# 主程序
# =========================

def main():

    end_date = datetime.now(timezone.utc).date()
    start_date = end_date - timedelta(days=SEARCH_DAYS)

    start = start_date.strftime("%Y/%m/%d")
    end = end_date.strftime("%Y/%m/%d")

    print("=" * 70)
    print("Weekly Literature Radar")
    print("=" * 70)

    print(f"Search period: {start} → {end}")
    print()

    all_papers = []

    for query in SEARCH_TERMS:

        print(f"Searching: {query}")

        pmids = search_pubmed(
            query,
            start,
            end
        )

        print(f"Found {len(pmids)} papers")

        papers = fetch_pubmed_articles(pmids)

        all_papers.extend(papers)

    all_papers = deduplicate(all_papers)

    print()
    print("=" * 70)
    print(f"Total unique papers: {len(all_papers)}")
    print("=" * 70)

    for i, paper in enumerate(all_papers, 1):

        print()
        print(f"[{i}] {paper['title']}")
        print(f"Journal: {paper['journal']}")
        print(f"Year: {paper['year']}")
        print(f"PMID: {paper['pmid']}")
        print(f"URL: {paper['url']}")

        if paper["abstract"]:
            print(
                "Abstract:",
                paper["abstract"][:500].replace("\n", " ")
            )


if __name__ == "__main__":
    main()
