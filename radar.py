import os
import re
import time
import json
import requests
from datetime import datetime, timedelta
from openai import OpenAI


# ============================================================
# Configuration
# ============================================================

SEARCH_DAYS = 7

# PubMed最多获取多少篇
MAX_RESULTS = 100

# 本地预筛选后，最多送给DeepSeek多少篇
MAX_AI_PAPERS = 40

# 最终输出多少篇
TOP_N = 15

# ============================================================
# Final score weights
# ============================================================

# 科学相关性必须占绝对主导
RELEVANCE_WEIGHT = 0.70

# 越新越优先
RECENCY_WEIGHT = 0.15

# 期刊层级
JOURNAL_WEIGHT = 0.15


# ============================================================
# PubMed query
# ============================================================

PUBMED_QUERY = """
(
    "transitive inference"[Title/Abstract]
    OR "transitive reasoning"[Title/Abstract]
    OR "relational inference"[Title/Abstract]
    OR "relational reasoning"[Title/Abstract]
    OR "relational memory"[Title/Abstract]
    OR "hierarchical representation"[Title/Abstract]
    OR "hierarchical structure"[Title/Abstract]
    OR "schema-based inference"[Title/Abstract]
    OR "schema-based reasoning"[Title/Abstract]
    OR (
        "hippocampus"[Title/Abstract]
        AND "prefrontal cortex"[Title/Abstract]
    )
    OR (
        "hippocampus"[Title/Abstract]
        AND "prefrontal"[Title/Abstract]
    )
    OR (
        "hippocampus"[Title/Abstract]
        AND "anterior cingulate"[Title/Abstract]
    )
    OR (
        "hippocampus"[Title/Abstract]
        AND "prelimbic"[Title/Abstract]
    )
    OR (
        "ventral hippocampus"[Title/Abstract]
        AND "prefrontal"[Title/Abstract]
    )
    OR (
        "CA1"[Title/Abstract]
        AND "prelimbic"[Title/Abstract]
    )
)
"""


# ============================================================
# Journal tiers
#
# These are PRIORITY TIERS, not automatically claimed JCR IFs.
# You can modify this list later.
# ============================================================

JOURNAL_TIERS = {

    # --------------------------------------------------------
    # Tier 1: Top neuroscience / cognition journals
    # --------------------------------------------------------

    "nature": 1,
    "science": 1,
    "cell": 1,

    "nature neuroscience": 1,
    "nature human behaviour": 1,
    "nature communications": 1,
    "nature methods": 1,

    "neuron": 1,
    "current biology": 1,
    "pnas": 1,
    "proceedings of the national academy of sciences": 1,

    "science advances": 1,
    "psychological science": 1,

    # --------------------------------------------------------
    # Tier 2: Strong field-specific journals
    # --------------------------------------------------------

    "journal of neuroscience": 2,
    "cerebral cortex": 2,
    "hippocampus": 2,
    "learning & memory": 2,
    "neurobiology of learning and memory": 2,
    "behavioral neuroscience": 2,
    "neuropsychologia": 2,
    "cognitive neuroscience": 2,
    "journal of cognitive neuroscience": 2,

    "european journal of neuroscience": 2,
    "cortex": 2,
    "cognition": 2,

    "frontiers in behavioral neuroscience": 2,
    "frontiers in neuroscience": 2,

    "european journal of psychology": 2,

    # --------------------------------------------------------
    # Tier 3: General / lower priority
    # --------------------------------------------------------

    "scientific reports": 3,
    "plos one": 3,
    "behavioral sciences": 3,

}


# ============================================================
# Core scientific keywords
# ============================================================

CORE_KEYWORDS = {

    "transitive inference": 12,
    "transitive reasoning": 12,

    "relational inference": 12,
    "relational reasoning": 11,

    "relational memory": 9,

    "hierarchical representation": 9,
    "hierarchical representations": 9,
    "hierarchical structure": 8,

    "schema-based inference": 10,
    "schema-based reasoning": 10,

    "structural knowledge": 7,

    "inferential": 5,
    "inference": 3,
}


# ============================================================
# Circuit keywords
# ============================================================

CIRCUIT_KEYWORDS = {

    "ventral hippocampus": 8,
    "vhippocampus": 8,
    "vca1": 9,

    "ca1": 5,
    "hippocampus": 4,

    "prelimbic": 8,
    "prelimbic cortex": 9,
    "pl cortex": 7,

    "prefrontal cortex": 5,
    "medial prefrontal cortex": 6,
    "mpfc": 6,

    "anterior cingulate": 7,
    "cingulate cortex": 5,
    "cg1": 8,

    "hippocampal-prefrontal": 9,
    "hippocampal prefrontal": 9,

    "hippocampus-prefrontal": 9,
    "hippocampus prefrontal": 9,

    "hippocampus-acc": 8,
    "hippocampal-acc": 8,
}


# ============================================================
# Methods / mechanisms
# ============================================================

METHOD_KEYWORDS = {

    "electrophysiology": 5,
    "single-unit": 4,
    "single unit": 4,

    "local field potential": 4,
    "lfp": 4,

    "theta": 3,
    "gamma": 3,

    "sharp-wave ripple": 4,
    "sharp wave ripple": 4,

    "population dynamics": 5,
    "neural dynamics": 4,
    "neural population": 4,

    "functional connectivity": 3,
    "coherence": 3,
    "granger": 3,

    "phase-amplitude coupling": 4,
    "phase amplitude coupling": 4,
    "pac": 3,

    "optogenetic": 5,
    "optogenetics": 5,

    "chemogenetic": 3,

    "projection-specific": 5,
    "projection specific": 5,

    "circuit manipulation": 5,
    "causal manipulation": 5,

    "computational model": 4,
    "computational modeling": 4,

    "reinforcement learning": 4,
    "value-based": 4,
    "value learning": 4,

    "schema": 3,
}


# ============================================================
# Low-priority / clinical topics
# ============================================================

LOW_PRIORITY_KEYWORDS = {

    "depression": -6,
    "schizophrenia": -6,

    "cancer": -8,

    "multiple sclerosis": -5,

    "substance use": -5,
    "cannabis use": -5,
    "alcohol": -4,
    "addiction": -5,

    "pain": -4,

    "trauma": -3,
    "post-traumatic": -3,

    "fear conditioning": -3,

    "stress": -2,

    "anxiety disorder": -5,

    "clinical": -4,
}


# ============================================================
# Utility
# ============================================================

def clean_text(text):
    """Normalize text for keyword matching."""

    if not text:
        return ""

    text = text.lower()
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def keyword_score(text, keyword_dict):

    score = 0

    for keyword, weight in keyword_dict.items():

        if keyword.lower() in text:
            score += weight

    return score


# ============================================================
# Local scientific relevance score
# ============================================================

def calculate_local_score(title, abstract):

    text = clean_text(
        f"{title} {abstract}"
    )

    core_score = keyword_score(
        text,
        CORE_KEYWORDS
    )

    circuit_score = keyword_score(
        text,
        CIRCUIT_KEYWORDS
    )

    method_score = keyword_score(
        text,
        METHOD_KEYWORDS
    )

    low_score = keyword_score(
        text,
        LOW_PRIORITY_KEYWORDS
    )

    score = (
        core_score
        + circuit_score
        + method_score
        + low_score
    )

    # --------------------------------------------------------
    # Detect important combinations
    # --------------------------------------------------------

    has_ti = any(
        k in text
        for k in [
            "transitive inference",
            "transitive reasoning",
            "relational inference",
            "relational reasoning",
        ]
    )

    has_hipp = any(
        k in text
        for k in [
            "hippocampus",
            "ventral hippocampus",
            "vca1",
            "ca1",
        ]
    )

    has_pfc = any(
        k in text
        for k in [
            "prefrontal",
            "prelimbic",
            "mpfc",
            "anterior cingulate",
            "cg1",
        ]
    )

    has_ephys = any(
        k in text
        for k in [
            "electrophysiology",
            "lfp",
            "local field potential",
            "single-unit",
            "single unit",
            "neural population",
            "population dynamics",
        ]
    )

    has_opto = any(
        k in text
        for k in [
            "optogenetic",
            "optogenetics",
            "projection-specific",
            "projection specific",
            "circuit manipulation",
        ]
    )

    # --------------------------------------------------------
    # Combination bonuses
    # --------------------------------------------------------

    if has_ti:
        score += 15

    if has_hipp and has_pfc:
        score += 8

    if has_hipp and has_pfc and has_ephys:
        score += 5

    if has_hipp and has_pfc and has_opto:
        score += 5

    if has_ti and has_hipp:
        score += 8

    return score


# ============================================================
# Publication year
# ============================================================

def extract_year(pub_date):

    if not pub_date:
        return None

    match = re.search(
        r"(19|20)\d{2}",
        pub_date
    )

    if match:
        return int(match.group())

    return None


# ============================================================
# Recency score
#
# 10 = current year
# 9  = previous year
# 8  = 2 years ago
# ...
# Older papers gradually decrease.
#
# Important:
# Old classic papers are NOT deleted.
# They simply receive less recency weight.
# ============================================================

def calculate_recency_score(pub_year):

    current_year = datetime.utcnow().year

    if not pub_year:
        return 5.0

    age = current_year - pub_year

    if age <= 0:
        return 10.0

    if age == 1:
        return 9.5

    if age == 2:
        return 9.0

    if age == 3:
        return 8.5

    if age == 4:
        return 8.0

    if age == 5:
        return 7.5

    if age <= 7:
        return 6.5

    if age <= 10:
        return 5.5

    if age <= 15:
        return 4.0

    if age <= 20:
        return 2.5

    # More than 20 years old
    return 1.0


# ============================================================
# Journal tier
# ============================================================

def normalize_journal_name(journal):

    if not journal:
        return ""

    journal = journal.lower()

    journal = journal.replace(
        ".",
        ""
    )

    journal = re.sub(
        r"\s+",
        " ",
        journal
    )

    return journal.strip()


def get_journal_tier(journal):

    normalized = normalize_journal_name(
        journal
    )

    if normalized in JOURNAL_TIERS:
        return JOURNAL_TIERS[normalized]

    # Partial matching
    for name, tier in JOURNAL_TIERS.items():

        if name in normalized:
            return tier

    # Unknown journal
    return 4


def journal_tier_score(journal):

    tier = get_journal_tier(
        journal
    )

    if tier == 1:
        return 10.0

    if tier == 2:
        return 8.0

    if tier == 3:
        return 5.0

    # Unknown / other journals
    return 4.0


# ============================================================
# PubMed
# ============================================================

def search_pubmed(
    query,
    days=7,
    max_results=100
):

    today = datetime.utcnow().date()

    start_date = (
        today - timedelta(days=days)
    )

    date_filter = (
        f'("{start_date.strftime("%Y/%m/%d")}"'
        f'[Date - Publication] : '
        f'"{today.strftime("%Y/%m/%d")}"'
        f'[Date - Publication])'
    )

    final_query = f"""
    ({query})
    AND
    {date_filter}
    """

    print("=" * 70)
    print("Weekly Literature Radar")
    print("=" * 70)

    print(
        f"Search period: "
        f"{start_date.strftime('%Y/%m/%d')} → "
        f"{today.strftime('%Y/%m/%d')}"
    )

    print("Searching PubMed...")

    search_url = (
        "https://eutils.ncbi.nlm.nih.gov/"
        "entrez/eutils/esearch.fcgi"
    )

    params = {
        "db": "pubmed",
        "term": final_query,
        "retmode": "json",
        "retmax": max_results,
        "sort": "pub date",
    }

    headers = {
        "User-Agent":
        "Weekly-Literature-Radar/1.0"
    }

    response = requests.get(
        search_url,
        params=params,
        headers=headers,
        timeout=30,
    )

    response.raise_for_status()

    data = response.json()

    pmids = (
        data
        .get("esearchresult", {})
        .get("idlist", [])
    )

    print(
        f"PubMed found {len(pmids)} papers."
    )

    if not pmids:
        return []

    # Avoid rapid consecutive requests
    time.sleep(0.5)

    return fetch_pubmed_articles(
        pmids
    )


# ============================================================
# Fetch PubMed article details
# ============================================================

def fetch_pubmed_articles(pmids):

    fetch_url = (
        "https://eutils.ncbi.nlm.nih.gov/"
        "entrez/eutils/efetch.fcgi"
    )

    params = {
        "db": "pubmed",
        "id": ",".join(pmids),
        "retmode": "xml",
    }

    headers = {
        "User-Agent":
        "Weekly-Literature-Radar/1.0"
    }

    response = requests.get(
        fetch_url,
        params=params,
        headers=headers,
        timeout=60,
    )

    response.raise_for_status()

    xml = response.text

    import xml.etree.ElementTree as ET

    root = ET.fromstring(xml)

    papers = []

    for article in root.findall(
        ".//PubmedArticle"
    ):

        pmid_node = article.find(
            ".//PMID"
        )

        if pmid_node is None:
            continue

        pmid = pmid_node.text.strip()

        # ----------------------------------------------------
        # Title
        # ----------------------------------------------------

        title_node = article.find(
            ".//ArticleTitle"
        )

        if title_node is not None:
            title = "".join(
                title_node.itertext()
            ).strip()
        else:
            title = ""

        # ----------------------------------------------------
        # Abstract
        # ----------------------------------------------------

        abstract_parts = []

        for abstract_text in article.findall(
            ".//Abstract/AbstractText"
        ):

            abstract_parts.append(
                "".join(
                    abstract_text.itertext()
                ).strip()
            )

        abstract = " ".join(
            abstract_parts
        )

        # ----------------------------------------------------
        # Journal
        # ----------------------------------------------------

        journal_node = article.find(
            ".//Journal/Title"
        )

        if (
            journal_node is not None
            and journal_node.text
        ):
            journal = (
                journal_node.text.strip()
            )
        else:
            journal = ""

        # ----------------------------------------------------
        # Publication date
        # ----------------------------------------------------

        pub_date = ""

        year_node = article.find(
            ".//PubDate/Year"
        )

        month_node = article.find(
            ".//PubDate/Month"
        )

        day_node = article.find(
            ".//PubDate/Day"
        )

        if (
            year_node is not None
            and year_node.text
        ):

            pub_date = year_node.text

            if (
                month_node is not None
                and month_node.text
            ):
                pub_date += (
                    f" {month_node.text}"
                )

            if (
                day_node is not None
                and day_node.text
            ):
                pub_date += (
                    f" {day_node.text}"
                )

        pub_year = extract_year(
            pub_date
        )

        # ----------------------------------------------------
        # URL
        # ----------------------------------------------------

        url = (
            "https://pubmed.ncbi.nlm.nih.gov/"
            f"{pmid}/"
        )

        papers.append(
            {
                "pmid": pmid,
                "title": title,
                "abstract": abstract,
                "journal": journal,
                "pub_date": pub_date,
                "pub_year": pub_year,
                "url": url,
            }
        )

    return papers


# ============================================================
# Local pre-filter
# ============================================================

def local_prefilter(papers):

    print("=" * 70)
    print("Local relevance pre-filter")
    print("=" * 70)

    scored = []

    for paper in papers:

        local_score = (
            calculate_local_score(
                paper["title"],
                paper["abstract"],
            )
        )

        recency_score = (
            calculate_recency_score(
                paper["pub_year"]
            )
        )

        journal_score = (
            journal_tier_score(
                paper["journal"]
            )
        )

        # ----------------------------------------------------
        # Preliminary score
        #
        # Scientific relevance remains dominant.
        # ----------------------------------------------------

        preliminary_score = (
            local_score * 0.70
            + recency_score * 0.15
            + journal_score * 0.15
        )

        paper["local_score"] = local_score
        paper["recency_score"] = recency_score
        paper["journal_score"] = journal_score
        paper["journal_tier"] = get_journal_tier(
            paper["journal"]
        )
        paper["preliminary_score"] = (
            preliminary_score
        )

        scored.append(paper)

    scored.sort(
        key=lambda x:
        x["preliminary_score"],
        reverse=True,
    )

    selected = scored[
        :MAX_AI_PAPERS
    ]

    print(
        f"Keeping {len(selected)} / "
        f"{len(papers)} papers "
        f"for DeepSeek analysis."
    )

    if selected:

        print(
            f"Preliminary score range: "
            f"{selected[-1]['preliminary_score']:.2f}"
            f" → "
            f"{selected[0]['preliminary_score']:.2f}"
        )

    return selected


# ============================================================
# DeepSeek analysis
# ============================================================

def analyze_with_deepseek(
    papers
):

    api_key = os.getenv(
        "OPENAI_API_KEY"
    )

    api_base = os.getenv(
        "OPENAI_API_BASE",
        "https://api.deepseek.com"
    )

    if not api_key:

        raise RuntimeError(
            "OPENAI_API_KEY is not configured."
        )

    client = OpenAI(
        api_key=api_key,
        base_url=api_base,
    )

    print("=" * 70)
    print("Sending papers to DeepSeek...")
    print("=" * 70)

    # --------------------------------------------------------
    # Build paper input
    # --------------------------------------------------------

    paper_texts = []

    for i, paper in enumerate(
        papers
    ):

        abstract = paper[
            "abstract"
        ]

        # Avoid unnecessarily large prompts
        if len(abstract) > 4000:
            abstract = (
                abstract[:4000]
                + "..."
            )

        paper_texts.append(
            f"""
PAPER {i + 1}

Title:
{paper["title"]}

Journal:
{paper["journal"]}

Publication year:
{paper["pub_year"]}

Publication date:
{paper["pub_date"]}

Abstract:
{abstract}

PubMed:
{paper["url"]}
"""
        )

    combined_text = "\n".join(
        paper_texts
    )

    # --------------------------------------------------------
    # System prompt
    # --------------------------------------------------------

    system_prompt = """
You are a scientific literature screening assistant for a neuroscience
researcher.

The researcher's main research interests are:

1. Transitive inference
2. Relational inference and relational reasoning
3. Relational memory
4. Hierarchical representations
5. Structural knowledge and schemas
6. Hippocampal-prefrontal interactions
7. Ventral CA1 (vCA1)
8. Prelimbic cortex (PL) / medial prefrontal cortex (mPFC)
9. Anterior cingulate cortex (ACC/Cg1)
10. Hippocampal-prefrontal and hippocampal-ACC circuit mechanisms
11. Electrophysiology, LFP, theta/gamma, ripples and neural population dynamics
12. Projection-specific circuit manipulation and optogenetics
13. Computational models of value-based learning versus relational representation

============================================================
IMPORTANT RELEVANCE PRINCIPLE
============================================================

Scientific relevance is more important than journal prestige.

A directly relevant transitive inference paper in a moderate journal
should rank above a prestigious but weakly related hippocampal paper.

Do NOT give a paper a high score simply because it contains:
"hippocampus", "prefrontal cortex", or "ACC".

============================================================
SCIENTIFIC RELEVANCE PRIORITY
============================================================

HIGHEST PRIORITY:

- Direct transitive inference
- Transitive reasoning
- Relational inference
- Relational reasoning
- Hierarchical representation
- Structural knowledge
- Schema-based inference
- Mechanistic studies of inference

HIGH PRIORITY:

- Hippocampal-prefrontal circuit mechanisms related to memory,
  inference, schemas, or relational representations
- vCA1-PL interactions
- hippocampus-mPFC interactions
- hippocampus-ACC interactions
- projection-specific manipulation of these circuits
- optogenetic causal tests of relevant circuits

MODERATE PRIORITY:

- Hippocampal-prefrontal electrophysiology
- theta/gamma coordination
- sharp-wave ripples
- neural population dynamics
- computational models related to relational or value-based learning

LOW PRIORITY:

- General hippocampal studies without inference or relational memory
- General PFC or ACC studies without memory/inference relevance
- Clinical studies where hippocampus/PFC is incidental
- Stress, depression, addiction, fear conditioning, etc.,
  unless they provide an unusually relevant circuit mechanism

============================================================
SCIENTIFIC RELEVANCE SCORE
============================================================

9.0-10.0:
Directly addresses transitive/relational inference or is an exceptionally
close mechanistic match to the researcher's main circuit question.

7.0-8.9:
Strongly relevant to hippocampal-prefrontal/ACC mechanisms,
relational memory, electrophysiology, or circuit manipulation.

5.0-6.9:
Useful background or methodological relevance, but not directly
addressing the central research questions.

3.0-4.9:
Indirectly related.

0-2.9:
Mostly irrelevant.

Be conservative.

============================================================
RECENCY
============================================================

Recency should be considered, but it must NOT override scientific relevance.

Very recent papers are preferred for the weekly radar.

However:
- Do NOT discard classic papers merely because they are old.
- A foundational paper can still be highly relevant.
- Papers older than 20 years should generally receive lower recency
  consideration unless they are clearly foundational.

============================================================
JOURNAL QUALITY
============================================================

Journal prestige is a secondary factor.

Prefer stronger field journals when scientific relevance is similar.

Do NOT allow journal prestige to compensate for poor scientific relevance.

For example:

A highly relevant paper in Journal of Neuroscience
should normally beat a generic hippocampal paper in Nature.

============================================================
OUTPUT
============================================================

For every paper provide:

- relevance_score: 0-10
- category
- why_relevant: one concise sentence
- tldr: one or two concise sentences

Do NOT calculate the final weighted score yourself.

The program will combine:
- scientific relevance
- recency
- journal tier

Return ONLY valid JSON.

Required format:

{
  "papers": [
    {
      "index": 1,
      "relevance_score": 9.5,
      "category": "Transitive inference",
      "why_relevant": "...",
      "tldr": "..."
    }
  ]
}
"""

    user_prompt = f"""
Please screen the following {len(papers)} papers.

Rank them according to their relevance to the researcher's specific
scientific interests.

Important:

1. Direct transitive inference should strongly outrank generic
   hippocampal studies.

2. Relational/hierarchical representation should outrank generic
   memory studies.

3. Mechanistic hippocampal-prefrontal/ACC circuit studies should
   outrank unrelated clinical hippocampal studies.

4. Recent papers should generally be preferred when scientific
   relevance is comparable.

5. Strong journals should generally be preferred when scientific
   relevance is comparable.

6. Do not give a high scientific relevance score simply because
   a paper contains hippocampus and prefrontal cortex.

Papers:

{combined_text}
"""

    # --------------------------------------------------------
    # API call
    # --------------------------------------------------------

    response = client.chat.completions.create(
        model="deepseek-flash",

        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ],

        response_format={
            "type": "json_object"
        },

        stream=False,
    )

    content = (
        response
        .choices[0]
        .message
        .content
    )

    if not content:

        raise RuntimeError(
            "DeepSeek returned an empty response."
        )

    try:

        result = json.loads(
            content
        )

    except json.JSONDecodeError as e:

        print(
            "Raw DeepSeek response:"
        )

        print(content)

        raise RuntimeError(
            f"DeepSeek returned invalid JSON: {e}"
        )

    return result


# ============================================================
# Merge AI results with metadata
# ============================================================

def merge_results(
    papers,
    ai_result
):

    ai_papers = ai_result.get(
        "papers",
        []
    )

    merged = []

    for item in ai_papers:

        index = item.get(
            "index"
        )

        if not isinstance(
            index,
            int
        ):
            continue

        if (
            index < 1
            or index > len(papers)
        ):
            continue

        paper = papers[
            index - 1
        ].copy()

        # ----------------------------------------------------
        # AI scientific relevance
        # ----------------------------------------------------

        try:

            relevance = float(
                item.get(
                    "relevance_score",
                    0
                )
            )

        except (
            TypeError,
            ValueError
        ):

            relevance = 0

        relevance = max(
            0,
            min(10, relevance)
        )

        # ----------------------------------------------------
        # Existing metadata
        # ----------------------------------------------------

        recency = paper.get(
            "recency_score",
            5.0
        )

        journal_score = paper.get(
            "journal_score",
            4.0
        )

        # ----------------------------------------------------
        # Final weighted score
        #
        # All components are 0-10.
        # ----------------------------------------------------

        final_score = (
            relevance * RELEVANCE_WEIGHT
            + recency * RECENCY_WEIGHT
            + journal_score * JOURNAL_WEIGHT
        )

        paper[
            "relevance_score"
        ] = relevance

        paper[
            "recency_score"
        ] = recency

        paper[
            "journal_score"
        ] = journal_score

        paper[
            "final_score"
        ] = final_score

        paper[
            "category"
        ] = item.get(
            "category",
            "Other"
        )

        paper[
            "why_relevant"
        ] = item.get(
            "why_relevant",
            ""
        )

        paper[
            "tldr"
        ] = item.get(
            "tldr",
            ""
        )

        merged.append(
            paper
        )

    # --------------------------------------------------------
    # Sort by final score
    # --------------------------------------------------------

    merged.sort(
        key=lambda x:
        x["final_score"],
        reverse=True,
    )

    return merged


# ============================================================
# Output
# ============================================================

def print_results(
    results
):

    print("=" * 70)
    print(
        f"TOP {min(TOP_N, len(results))} PAPERS"
    )
    print("=" * 70)

    for i, paper in enumerate(
        results[:TOP_N],
        start=1
    ):

        print(
            f"[{i}] "
            f"{paper['final_score']:.1f}/10 "
            f"{paper['title']}"
        )

        print(
            f"Relevance: "
            f"{paper['relevance_score']:.1f}/10"
        )

        print(
            f"Recency: "
            f"{paper['recency_score']:.1f}/10"
        )

        print(
            f"Journal: "
            f"{paper['journal']} "
            f"(Tier {paper['journal_tier']})"
        )

        print(
            f"Publication year: "
            f"{paper['pub_year']}"
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

        print()


# ============================================================
# Main
# ============================================================

def main():

    # --------------------------------------------------------
    # 1. PubMed search
    # --------------------------------------------------------

    papers = search_pubmed(
        PUBMED_QUERY,
        days=SEARCH_DAYS,
        max_results=MAX_RESULTS,
    )

    if not papers:

        print(
            "No papers found."
        )

        return

    # --------------------------------------------------------
    # 2. Deduplicate
    # --------------------------------------------------------

    unique = {}

    for paper in papers:

        unique[
            paper["pmid"]
        ] = paper

    papers = list(
        unique.values()
    )

    print(
        f"Total unique papers: "
        f"{len(papers)}"
    )

    # --------------------------------------------------------
    # 3. Local pre-filter
    # --------------------------------------------------------

    selected_papers = (
        local_prefilter(
            papers
        )
    )

    if not selected_papers:

        print(
            "No papers passed "
            "local filtering."
        )

        return

    # --------------------------------------------------------
    # 4. DeepSeek screening
    # --------------------------------------------------------

    ai_result = (
        analyze_with_deepseek(
            selected_papers
        )
    )

    # --------------------------------------------------------
    # 5. Merge
    # --------------------------------------------------------

    results = (
        merge_results(
            selected_papers,
            ai_result
        )
    )

    # --------------------------------------------------------
    # 6. Output
    # --------------------------------------------------------

    print_results(
        results
    )


if __name__ == "__main__":
    main()
