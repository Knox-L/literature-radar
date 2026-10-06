import time
import requests
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone


# =========================
# 基本设置
# =========================

SEARCH_DAYS = 7
MAX_RESULTS = 100

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


# =========================
# PubMed 请求
# =========================

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

    raise RuntimeError("PubMed request failed after retries.")


# =========================
# 搜索 PubMed
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
        "sort": "pub date",
    }

    response = pubmed_request(url, params)

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

    response = pubmed_request(url, params)

    root = ET.fromstring(response.text)

    papers = []

    for article in root.findall(".//PubmedArticle"):

        title_node = article.find(".//ArticleTitle")

        title = ""

        if title_node is not None:
            title = "".join(
                title_node.itertext()
            ).strip()

        # -------------------------
        # Abstract
        # -------------------------

        abstract_parts = []

        for node in article.findall(".//Abstract/AbstractText"):

            text = "".join(node.itertext()).strip()

            if text:
                label = node.attrib.get("Label")

                if label:
                    abstract_parts.append(
                        f"{label}: {text}"
                    )
                else:
                    abstract_parts.append(text)

        abstract = " ".join(abstract_parts)

        # -------------------------
        # Journal
        # -------------------------

        journal = article.findtext(
            ".//Journal/Title",
            default=""
        )

        # -------------------------
        # Publication year
        # -------------------------

        year = (
            article.findtext(".//PubDate/Year")
            or article.findtext(".//PubDate/MedlineDate")
            or ""
        )

        # -------------------------
        # PMID
        # -------------------------

        pmid = article.findtext(
            ".//PMID",
            default=""
        )

        # -------------------------
        # DOI
        # -------------------------

        doi = ""

        for article_id in article.findall(".//ArticleId"):

            if article_id.attrib.get("IdType") == "doi":

                doi = article_id.text or ""

                break

        # -------------------------
        # Authors
        # -------------------------

        authors = []

        for author in article.findall(".//Author"):

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
                    name += f" {initials}"

                authors.append(name)

        papers.append({
            "title": title,
            "abstract": abstract,
            "journal": journal,
            "year": year,
            "pmid": pmid,
            "doi": doi,
            "authors": authors,
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

        key = (
            paper["pmid"]
            or paper["doi"]
            or paper["title"].lower()
        )

        if key not in seen:

            seen.add(key)

            unique.append(paper)

    return unique


# =========================
# 主程序
# =========================

def main():

    end_date = datetime.now(
        timezone.utc
    ).date()

    start_date = (
        end_date -
        timedelta(days=SEARCH_DAYS)
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

    print()
    print("Searching PubMed...")
    print()

    # -------------------------
    # Search
    # -------------------------

    pmids = search_pubmed(
        PUBMED_QUERY,
        start,
        end
    )

    print(
        f"PubMed found {len(pmids)} papers."
    )

    # -------------------------
    # Fetch
    # -------------------------

    papers = fetch_pubmed_articles(
        pmids
    )

    # -------------------------
    # Deduplicate
    # -------------------------

    papers = deduplicate(papers)

    print()
    print("=" * 70)

    print(
        f"Total unique papers: {len(papers)}"
    )

    print("=" * 70)

    # -------------------------
    # Display
    # -------------------------

    for i, paper in enumerate(
        papers,
        1
    ):

        print()

        print(
            f"[{i}] {paper['title']}"
        )

        print(
            f"Journal: {paper['journal']}"
        )

        print(
            f"Year: {paper['year']}"
        )

        print(
            f"PMID: {paper['pmid']}"
        )

        if paper["doi"]:

            print(
                f"DOI: {paper['doi']}"
            )

        print(
            f"URL: {paper['url']}"
        )

        if paper["authors"]:

            print(
                "Authors:",
                ", ".join(
                    paper["authors"][:5]
                )
            )

        if paper["abstract"]:

            abstract_preview = (
                paper["abstract"]
                .replace("\n", " ")
            )

            print(
                "Abstract:",
                abstract_preview[:600]
            )


if __name__ == "__main__":

    main()
