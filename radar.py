import os
import json
import re
import smtplib
from datetime import datetime, timedelta
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

import requests
from openai import OpenAI


# ============================================================
# Configuration
# ============================================================

SEARCH_DAYS = 7
MAX_RESULTS = 100
MAX_AI_PAPERS = 40
TOP_N = 15

# Papers with AI relevance below this score will not be emailed
MIN_RELEVANCE_SCORE = 3.0

# Final score weights
RELEVANCE_WEIGHT = 0.70
RECENCY_WEIGHT = 0.15
JOURNAL_WEIGHT = 0.15

# File used to store successfully emailed PMIDs
SEEN_FILE = "seen_pmids.json"

# 163 SMTP settings
SMTP_SERVER = "smtp.163.com"
SMTP_PORT = 465


# ============================================================
# Research interests
# ============================================================

CORE_TI_KEYWORDS = [
    "transitive inference",
    "transitive reasoning",
    "transitive relationship",
    "transitive relationships",
    "transitive learning",
]

CORE_RELATIONAL_KEYWORDS = [
    "relational inference",
    "relational reasoning",
    "relational memory",
    "relational representation",
    "relational representations",
    "hierarchical representation",
    "hierarchical representations",
    "hierarchical structure",
    "hierarchical organization",
    "schema-based inference",
    "schema based inference",
    "schema-based reasoning",
    "schema based reasoning",
    "structural knowledge",
    "relational knowledge",
]

CIRCUIT_KEYWORDS = [
    "hippocamp",
    "prefrontal",
    "prefrontal cortex",
    "mPFC",
    "medial prefrontal",
    "prelimbic",
    "prelimbic cortex",
    "PL cortex",
    "anterior cingulate",
    "ACC",
    "cingulate cortex",
    "vCA1",
    "ventral CA1",
    "CA1",
    "hippocampal-prefrontal",
    "hippocampal prefrontal",
    "hippocampal-cortical",
    "hippocampal cortical",
]

MECHANISM_KEYWORDS = [
    "electrophysiology",
    "electrophysiological",
    "single-unit",
    "single neuron",
    "neuronal dynamics",
    "neural dynamics",
    "local field potential",
    "LFP",
    "theta",
    "gamma",
    "sharp-wave ripple",
    "sharp wave ripple",
    "ripple",
    "phase-amplitude coupling",
    "PAC",
    "coherence",
    "connectivity",
    "functional connectivity",
    "effective connectivity",
    "granger",
    "causal",
    "optogenetic",
    "optogenetics",
    "chemogenetic",
    "neural decoding",
    "population coding",
]

GENERAL_MEMORY_KEYWORDS = [
    "memory",
    "learning",
    "cognitive",
    "cognition",
    "decision making",
    "decision-making",
    "schema",
    "spatial memory",
    "episodic memory",
]

LOW_PRIORITY_KEYWORDS = [
    "depression",
    "anxiety",
    "schizophrenia",
    "addiction",
    "multiple sclerosis",
    "stroke",
    "aphasia",
    "clinical",
    "patient",
    "psychopathology",
]


# ============================================================
# Journal tiers
# ============================================================

JOURNAL_TIER_1 = {
    "Nature",
    "Science",
    "Cell",
    "Nature Neuroscience",
    "Nature Human Behaviour",
    "Nature Communications",
    "Neuron",
    "Current Biology",
    "Proceedings of the National Academy of Sciences",
    "PNAS",
    "Science Advances",
    "Psychological Science",
}

JOURNAL_TIER_2 = {
    "Journal of Neuroscience",
    "Cerebral Cortex",
    "Hippocampus",
    "Learning & Memory",
    "Neurobiology of Learning and Memory",
    "Neuropsychologia",
    "Cognitive Neuroscience",
    "Journal of Cognitive Neuroscience",
    "eLife",
    "Behavioral Neuroscience",
}

JOURNAL_TIER_3 = {
    "Scientific Reports",
    "PLOS ONE",
    "Behavioral Sciences",
}

JOURNAL_TIER_4 = {
    "NeuroImage",
    "Journal of Neurophysiology",
    "eNeuro",
    "Cortex",
}


# ============================================================
# PubMed
# ============================================================

PUBMED_EUTILS = (
    "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
)


def build_pubmed_query():
    """
    Research-focused PubMed query.
    """

    return """
    (
        "transitive inference"[Title/Abstract]
        OR "transitive reasoning"[Title/Abstract]
        OR "transitive relationship"[Title/Abstract]
        OR "transitive relationships"[Title/Abstract]
        OR "relational inference"[Title/Abstract]
        OR "relational reasoning"[Title/Abstract]
        OR "relational memory"[Title/Abstract]
        OR "relational representation"[Title/Abstract]
        OR "relational representations"[Title/Abstract]
        OR "hierarchical representation"[Title/Abstract]
        OR "hierarchical representations"[Title/Abstract]
        OR "hierarchical structure"[Title/Abstract]
        OR "schema-based inference"[Title/Abstract]
        OR "schema-based reasoning"[Title/Abstract]
        OR "structural knowledge"[Title/Abstract]
        OR "structural learning"[Title/Abstract]

        OR
        (
            hippocamp*[Title/Abstract]
            AND
            (
                prefrontal[Title/Abstract]
                OR "prefrontal cortex"[Title/Abstract]
                OR mPFC[Title/Abstract]
                OR prelimbic[Title/Abstract]
                OR "anterior cingulate"[Title/Abstract]
                OR ACC[Title/Abstract]
                OR "ventral CA1"[Title/Abstract]
                OR vCA1[Title/Abstract]
            )
        )
    )
    """


def pubmed_search(query, days=7, max_results=100):
    """
    Search PubMed within the recent publication window.
    """

    end_date = datetime.utcnow().date()
    start_date = end_date - timedelta(days=days)

    date_filter = (
        f'("{start_date.strftime("%Y/%m/%d")}"'
        f'[Date - Publication] : '
        f'"{end_date.strftime("%Y/%m/%d")}"'
        f'[Date - Publication])'
    )

    full_query = f"({query}) AND {date_filter}"

    params = {
        "db": "pubmed",
        "term": full_query,
        "retmode": "json",
        "retmax": max_results,
        "sort": "date",
    }

    response = requests.get(
        f"{PUBMED_EUTILS}/esearch.fcgi",
        params=params,
        timeout=30,
    )

    response.raise_for_status()

    data = response.json()

    return data["esearchresult"]["idlist"]


def clean_xml_text(text):
    """
    Basic cleanup of PubMed XML text.
    """

    text = re.sub(r"<[^>]+>", "", text)

    text = (
        text.replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", '"')
        .replace("&#39;", "'")
    )

    return text.strip()


def fetch_pubmed_details(pmids):
    """
    Fetch title, abstract, journal, year and PMID.
    """

    if not pmids:
        return []

    params = {
        "db": "pubmed",
        "id": ",".join(pmids),
        "retmode": "xml",
    }

    response = requests.get(
        f"{PUBMED_EUTILS}/efetch.fcgi",
        params=params,
        timeout=60,
    )

    response.raise_for_status()

    xml = response.text

    papers = []

    articles = re.findall(
        r"<PubmedArticle>(.*?)</PubmedArticle>",
        xml,
        flags=re.DOTALL,
    )

    for article in articles:

        pmid_match = re.search(
            r"<PMID[^>]*>(.*?)</PMID>",
            article,
            flags=re.DOTALL,
        )

        title_match = re.search(
            r"<ArticleTitle>(.*?)</ArticleTitle>",
            article,
            flags=re.DOTALL,
        )

        journal_match = re.search(
            r"<Journal>.*?<Title>(.*?)</Title>",
            article,
            flags=re.DOTALL,
        )

        abstract_matches = re.findall(
            r"<AbstractText[^>]*>(.*?)</AbstractText>",
            article,
            flags=re.DOTALL,
        )

        year_match = re.search(
            r"<PubDate>.*?<Year>(.*?)</Year>",
            article,
            flags=re.DOTALL,
        )

        if not year_match:
            year_match = re.search(
                r"<PubDate>.*?<MedlineDate>(.*?)</MedlineDate>",
                article,
                flags=re.DOTALL,
            )

        pmid = (
            clean_xml_text(pmid_match.group(1))
            if pmid_match
            else ""
        )

        title = (
            clean_xml_text(title_match.group(1))
            if title_match
            else "Unknown title"
        )

        journal = (
            clean_xml_text(journal_match.group(1))
            if journal_match
            else "Unknown journal"
        )

        abstract_parts = [
            clean_xml_text(x)
            for x in abstract_matches
        ]

        abstract = " ".join(abstract_parts)

        year_text = (
            clean_xml_text(year_match.group(1))
            if year_match
            else str(datetime.utcnow().year)
        )

        year_match_num = re.search(
            r"(20\d{2}|19\d{2})",
            year_text,
        )

        year = (
            int(year_match_num.group(1))
            if year_match_num
            else datetime.utcnow().year
        )

        papers.append(
            {
                "pmid": pmid,
                "title": title,
                "journal": journal,
                "abstract": abstract,
                "year": year,
            }
        )

    return papers


# ============================================================
# De-duplication
# ============================================================

def load_seen_pmids():
    """
    Load PMIDs that have already been successfully emailed.
    """

    if not os.path.exists(SEEN_FILE):
        return set()

    try:
        with open(
            SEEN_FILE,
            "r",
            encoding="utf-8",
        ) as f:

            data = json.load(f)

        if isinstance(data, list):
            return {
                str(x)
                for x in data
            }

        return set()

    except Exception as e:

        print(
            "Warning: Could not read "
            f"{SEEN_FILE}: {e}"
        )

        return set()


def save_seen_pmids(pmids):
    """
    Save successfully emailed PMIDs.
    """

    with open(
        SEEN_FILE,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            sorted(list(pmids)),
            f,
            ensure_ascii=False,
            indent=2,
        )


def remove_seen_papers(
    papers,
    seen_pmids,
):
    """
    Remove papers that have already been emailed.
    """

    new_papers = [
        paper
        for paper in papers
        if paper["pmid"] not in seen_pmids
    ]

    already_sent = (
        len(papers) - len(new_papers)
    )

    print(
        f"Found {len(papers)} papers; "
        f"{already_sent} already sent; "
        f"{len(new_papers)} new papers remain."
    )

    return new_papers


# ============================================================
# Local relevance scoring
# ============================================================

def count_keyword_hits(
    text,
    keywords,
):
    text_lower = text.lower()

    return sum(
        1
        for keyword in keywords
        if keyword.lower() in text_lower
    )


def local_scientific_score(paper):
    """
    Hierarchical local relevance score.

    Priority:
    1. Transitive inference
    2. Relational inference / hierarchical representation
    3. Hippocampal-prefrontal / ACC circuit
    4. Neural mechanisms
    5. General memory/cognition
    """

    text = (
        paper["title"]
        + " "
        + paper["abstract"]
    )

    ti_hits = count_keyword_hits(
        text,
        CORE_TI_KEYWORDS,
    )

    relational_hits = count_keyword_hits(
        text,
        CORE_RELATIONAL_KEYWORDS,
    )

    circuit_hits = count_keyword_hits(
        text,
        CIRCUIT_KEYWORDS,
    )

    mechanism_hits = count_keyword_hits(
        text,
        MECHANISM_KEYWORDS,
    )

    memory_hits = count_keyword_hits(
        text,
        GENERAL_MEMORY_KEYWORDS,
    )

    low_priority_hits = count_keyword_hits(
        text,
        LOW_PRIORITY_KEYWORDS,
    )

    score = 0

    score += min(ti_hits, 3) * 10
    score += min(relational_hits, 4) * 6
    score += min(circuit_hits, 5) * 3
    score += min(mechanism_hits, 5) * 2
    score += min(memory_hits, 3) * 1

    score -= min(low_priority_hits, 3) * 4

    # Strong combinations
    if ti_hits > 0 and circuit_hits > 0:
        score += 15

    if relational_hits > 0 and circuit_hits > 0:
        score += 10

    if (
        (ti_hits > 0 or relational_hits > 0)
        and mechanism_hits > 0
    ):
        score += 8

    return max(score, 0)


# ============================================================
# Recency
# ============================================================

def recency_score(year):

    current_year = datetime.utcnow().year

    age = current_year - year

    if age <= 0:
        return 10.0

    elif age == 1:
        return 9.5

    elif age == 2:
        return 9.0

    elif age == 3:
        return 8.5

    elif age == 4:
        return 8.0

    elif age == 5:
        return 7.5

    elif age <= 7:
        return 6.5

    elif age <= 10:
        return 5.5

    elif age <= 15:
        return 4.0

    elif age <= 20:
        return 2.5

    else:
        return 1.0


# ============================================================
# Journal
# ============================================================

def normalize_journal_name(journal):
    return journal.strip().lower()


def journal_score(journal):

    journal_lower = normalize_journal_name(
        journal
    )

    tier_1 = {
        normalize_journal_name(x)
        for x in JOURNAL_TIER_1
    }

    tier_2 = {
        normalize_journal_name(x)
        for x in JOURNAL_TIER_2
    }

    tier_3 = {
        normalize_journal_name(x)
        for x in JOURNAL_TIER_3
    }

    tier_4 = {
        normalize_journal_name(x)
        for x in JOURNAL_TIER_4
    }

    if journal_lower in tier_1:
        return 10.0, "Tier 1"

    if journal_lower in tier_2:
        return 8.0, "Tier 2"

    if journal_lower in tier_3:
        return 5.0, "Tier 3"

    if journal_lower in tier_4:
        return 4.0, "Tier 4"

    return 4.0, "Unknown"


# ============================================================
# DeepSeek
# ============================================================

def create_client():

    api_key = os.environ.get(
        "OPENAI_API_KEY"
    )

    api_base = os.environ.get(
        "OPENAI_API_BASE"
    )

    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set."
        )

    if not api_base:
        raise RuntimeError(
            "OPENAI_API_BASE is not set."
        )

    return OpenAI(
        api_key=api_key,
        base_url=api_base,
    )


def ai_score_papers(papers):

    client = create_client()

    paper_text = []

    for i, paper in enumerate(papers):

        paper_text.append(
            f"""
INDEX: {i}

TITLE:
{paper["title"]}

JOURNAL:
{paper["journal"]}

YEAR:
{paper["year"]}

ABSTRACT:
{paper["abstract"]}
"""
        )

    combined_text = "\n".join(
        paper_text
    )

    system_prompt = """
You are a scientific literature relevance evaluator.

The researcher's main interests are:

1. Transitive inference
2. Relational inference and relational reasoning
3. Relational memory and hierarchical representation
4. Hippocampal-prefrontal/ACC mechanisms
5. Hippocampus, especially CA1/vCA1
6. Prelimbic/mPFC and ACC/Cg1
7. Electrophysiology, LFP, neural dynamics and inter-regional coordination
8. Optogenetic/causal circuit manipulation
9. Computational models of value-based versus relational inference

Use the following relevance hierarchy.

10 = directly studies transitive inference or a closely related inference paradigm and is highly relevant.

9 = directly studies relational inference/reasoning, hierarchical representations, or schema-based inference and is strongly relevant.

8 = directly studies hippocampal-prefrontal/ACC mechanisms of relational memory, inference, learning, or hierarchical representations.

7 = studies neural mechanisms highly relevant to the researcher's circuit/electrophysiological interests, even if not directly about transitive inference.

5-6 = moderately relevant cognitive or memory research with some useful conceptual or neural overlap.

3-4 = general hippocampal, prefrontal, memory, learning, or decision-making research with limited direct relevance.

1-2 = peripheral relevance.

0 = essentially irrelevant.

Important rules:

- Scientific relevance is more important than journal prestige.
- A Nature/Science paper should NOT receive a high relevance score merely because of journal prestige.
- Direct transitive inference and relational inference papers should rank above generic hippocampus/PFC papers.
- Papers about clinical disorders should generally receive low scores unless they directly address relational inference or the relevant circuit mechanism.
- Artificial intelligence papers about relational reasoning should generally receive low scores unless they provide a directly useful computational theory for biological transitive inference.
- Prefer rat/mouse neuroscience studies when otherwise comparable.
- Prefer hippocampus-PL/ACC/vCA1 circuit studies when otherwise comparable.
- Prefer electrophysiology, neural dynamics, and causal manipulation when otherwise comparable.

Return ONLY valid JSON.

Format:

[
  {
    "index": 0,
    "relevance_score": 9.5,
    "category": "Transitive inference",
    "why_relevant": "One concise sentence.",
    "tldr": "One concise sentence."
  }
]

Do not include markdown.
"""

    response = client.chat.completions.create(
        model="deepseek-flash",
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": combined_text,
            },
        ],
        temperature=0,
    )

    content = (
        response.choices[0]
        .message.content
        .strip()
    )

    # Remove accidental Markdown fences
    content = re.sub(
        r"^```json\s*",
        "",
        content,
        flags=re.IGNORECASE,
    )

    content = re.sub(
        r"\s*```$",
        "",
        content,
    )

    return json.loads(content)


# ============================================================
# Combine scores
# ============================================================

def combine_scores(
    papers,
    ai_results,
):

    ai_by_index = {
        item["index"]: item
        for item in ai_results
    }

    scored = []

    for i, paper in enumerate(papers):

        if i not in ai_by_index:
            continue

        ai = ai_by_index[i]

        relevance = float(
            ai.get(
                "relevance_score",
                0,
            )
        )

        recency = recency_score(
            paper["year"]
        )

        journal, tier = journal_score(
            paper["journal"]
        )

        final_score = (
            RELEVANCE_WEIGHT * relevance
            + RECENCY_WEIGHT * recency
            + JOURNAL_WEIGHT * journal
        )

        paper_copy = paper.copy()

        paper_copy.update(
            {
                "relevance_score": relevance,
                "recency_score": recency,
                "journal_score": journal,
                "journal_tier": tier,
                "final_score": final_score,
                "category": ai.get(
                    "category",
                    "Other",
                ),
                "why_relevant": ai.get(
                    "why_relevant",
                    "",
                ),
                "tldr": ai.get(
                    "tldr",
                    "",
                ),
            }
        )

        scored.append(
            paper_copy
        )

    # Do not fill TOP_N with irrelevant papers
    scored = [
        paper
        for paper in scored
        if paper["relevance_score"]
        >= MIN_RELEVANCE_SCORE
    ]

    scored.sort(
        key=lambda x: x["final_score"],
        reverse=True,
    )

    return scored[:TOP_N]


# ============================================================
# Email
# ============================================================

def create_email_html(papers):

    today = datetime.now().strftime(
        "%Y-%m-%d"
    )

    html = f"""
<html>
<head>
<meta charset="UTF-8">
</head>

<body>

<h2>Weekly Literature Radar</h2>

<p>
<strong>Date:</strong> {today}<br>
<strong>New relevant papers:</strong> {len(papers)}
</p>

<hr>
"""

    for rank, paper in enumerate(
        papers,
        1,
    ):

        pubmed_url = (
            "https://pubmed.ncbi.nlm.nih.gov/"
            f"{paper['pmid']}/"
        )

        html += f"""
<h3>
{rank}. {paper['title']}
</h3>

<p>
<strong>Final score:</strong>
{paper['final_score']:.1f}/10
<br>

<strong>Relevance:</strong>
{paper['relevance_score']:.1f}/10
<br>

<strong>Journal:</strong>
{paper['journal']}
({paper['journal_tier']})
<br>

<strong>Year:</strong>
{paper['year']}
<br>

<strong>Category:</strong>
{paper['category']}
</p>

<p>
<strong>Why relevant:</strong>
{paper['why_relevant']}
</p>

<p>
<strong>TL;DR:</strong>
{paper['tldr']}
</p>

<p>
<a href="{pubmed_url}">
View on PubMed
</a>
</p>

<hr>
"""

    html += """
<p>
This literature radar was automatically generated.
</p>

</body>
</html>
"""

    return html


def send_email(papers):

    sender = os.environ.get(
        "SENDER"
    )

    receiver = os.environ.get(
        "RECEIVER"
    )

    password = os.environ.get(
        "SENDER_PASSWORD"
    )

    if not sender:
        raise RuntimeError(
            "SENDER is not set."
        )

    if not receiver:
        raise RuntimeError(
            "RECEIVER is not set."
        )

    if not password:
        raise RuntimeError(
            "SENDER_PASSWORD is not set."
        )

    today = datetime.now().strftime(
        "%Y-%m-%d"
    )

    subject = (
        f"Weekly Literature Radar — "
        f"{today} — {len(papers)} papers"
    )

    message = MIMEMultipart(
        "alternative"
    )

    message["From"] = sender
    message["To"] = receiver
    message["Subject"] = subject

    html = create_email_html(
        papers
    )

    message.attach(
        MIMEText(
            html,
            "html",
            "utf-8",
        )
    )

    print(
        f"Sending email from {sender} "
        f"to {receiver}..."
    )

    with smtplib.SMTP_SSL(
        SMTP_SERVER,
        SMTP_PORT,
        timeout=30,
    ) as server:

        server.login(
            sender,
            password,
        )

        server.sendmail(
            sender,
            receiver,
            message.as_string(),
        )

    print(
        "Email sent successfully."
    )


# ============================================================
# Main
# ============================================================

def main():

    print(
        "Starting weekly literature radar..."
    )

    # --------------------------------------------------------
    # 1. Load history
    # --------------------------------------------------------

    seen_pmids = load_seen_pmids()

    print(
        f"Previously sent papers: "
        f"{len(seen_pmids)}"
    )

    # --------------------------------------------------------
    # 2. Search PubMed
    # --------------------------------------------------------

    query = build_pubmed_query()

    pmids = pubmed_search(
        query,
        days=SEARCH_DAYS,
        max_results=MAX_RESULTS,
    )

    print(
        f"PubMed returned "
        f"{len(pmids)} papers."
    )

    # --------------------------------------------------------
    # 3. Fetch details
    # --------------------------------------------------------

    papers = fetch_pubmed_details(
        pmids
    )

    print(
        f"Successfully fetched "
        f"{len(papers)} papers."
    )

    # --------------------------------------------------------
    # 4. Remove already sent papers
    # --------------------------------------------------------

    papers = remove_seen_papers(
        papers,
        seen_pmids,
    )

    if not papers:

        print(
            "No new papers found."
        )

        return

    # --------------------------------------------------------
    # 5. Local pre-filter
    # --------------------------------------------------------

    for paper in papers:

        paper["local_score"] = (
            local_scientific_score(
                paper
            )
        )

        paper["local_recency"] = (
            recency_score(
                paper["year"]
            )
        )

        (
            paper["local_journal"],
            _
        ) = journal_score(
            paper["journal"]
        )

        paper["preliminary_score"] = (
            0.70
            * paper["local_score"]
            + 0.15
            * paper["local_recency"]
            + 0.15
            * paper["local_journal"]
        )

    papers.sort(
        key=lambda x: x["preliminary_score"],
        reverse=True,
    )

    papers_for_ai = papers[
        :MAX_AI_PAPERS
    ]

    print(
        f"Sending top "
        f"{len(papers_for_ai)} "
        f"candidates to DeepSeek..."
    )

    # --------------------------------------------------------
    # 6. AI scoring
    # --------------------------------------------------------

    ai_results = ai_score_papers(
        papers_for_ai
    )

    print(
        f"DeepSeek evaluated "
        f"{len(ai_results)} papers."
    )

    # --------------------------------------------------------
    # 7. Combine scores
    # --------------------------------------------------------

    final_papers = combine_scores(
        papers_for_ai,
        ai_results,
    )

    # --------------------------------------------------------
    # 8. No relevant papers
    # --------------------------------------------------------

    if not final_papers:

        print(
            "No papers met the minimum "
            "relevance threshold."
        )

        return

    # --------------------------------------------------------
    # 9. Print results
    # --------------------------------------------------------

    print("\n")
    print("=" * 80)
    print(
        "FINAL LITERATURE RADAR RESULTS"
    )
    print("=" * 80)

    for rank, paper in enumerate(
        final_papers,
        1,
    ):

        print("\n")

        print(
            f"{rank}. "
            f"{paper['final_score']:.1f}/10 — "
            f"{paper['title']}"
        )

        print(
            f"   Relevance: "
            f"{paper['relevance_score']:.1f}"
        )

        print(
            f"   Recency: "
            f"{paper['recency_score']:.1f}"
        )

        print(
            f"   Journal: "
            f"{paper['journal']} "
            f"({paper['journal_tier']})"
        )

        print(
            f"   Year: "
            f"{paper['year']}"
        )

        print(
            f"   Category: "
            f"{paper['category']}"
        )

        print(
            f"   Why relevant: "
            f"{paper['why_relevant']}"
        )

        print(
            f"   TL;DR: "
            f"{paper['tldr']}"
        )

        print(
            f"   PubMed: "
            f"https://pubmed.ncbi.nlm.nih.gov/"
            f"{paper['pmid']}/"
        )

    # --------------------------------------------------------
    # 10. Send email
    # --------------------------------------------------------

    print("\n")
    print(
        "Preparing email..."
    )

    send_email(
        final_papers
    )

    # --------------------------------------------------------
    # 11. ONLY AFTER successful email:
    #     save PMIDs as sent
    # --------------------------------------------------------

    newly_sent_pmids = {
        paper["pmid"]
        for paper in final_papers
    }

    seen_pmids.update(
        newly_sent_pmids
    )

    save_seen_pmids(
        seen_pmids
    )

    print(
        f"Recorded "
        f"{len(newly_sent_pmids)} "
        f"newly sent PMIDs."
    )

    print("\n")
    print(
        "Literature radar completed successfully."
    )


if __name__ == "__main__":
    main()
