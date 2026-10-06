import os
import json
import re
import requests
from datetime import datetime, timedelta
from openai import OpenAI


# ============================================================
# Basic settings
# ============================================================

SEARCH_DAYS = 7
MAX_RESULTS = 100
MAX_AI_PAPERS = 40
TOP_N = 15
MIN_RELEVANCE_SCORE = 3.0

# Final score:
# scientific relevance = 70%
# recency = 15%
# journal quality = 15%
RELEVANCE_WEIGHT = 0.70
RECENCY_WEIGHT = 0.15
JOURNAL_WEIGHT = 0.15


# ============================================================
# PubMed search terms
# ============================================================

PUBMED_QUERY = """
(
    "transitive inference"[Title/Abstract]
    OR "transitive reasoning"[Title/Abstract]
    OR "relational inference"[Title/Abstract]
    OR "relational reasoning"[Title/Abstract]
    OR "relational memory"[Title/Abstract]
    OR "relational representation"[Title/Abstract]
    OR "relational structure"[Title/Abstract]
    OR "hierarchical representation"[Title/Abstract]
    OR "hierarchical structure"[Title/Abstract]
    OR "schema-based inference"[Title/Abstract]
    OR "schema-based reasoning"[Title/Abstract]
    OR "structural learning"[Title/Abstract]
    OR "structure learning"[Title/Abstract]
    OR
    (
        hippocamp*[Title/Abstract]
        AND
        (
            prefrontal[Title/Abstract]
            OR mPFC[Title/Abstract]
            OR prelimbic[Title/Abstract]
            OR "prelimbic cortex"[Title/Abstract]
            OR "anterior cingulate"[Title/Abstract]
            OR ACC[Title/Abstract]
            OR Cg1[Title/Abstract]
        )
    )
)
"""


# ============================================================
# Scientific relevance keywords
# ============================================================

# ------------------------------------------------------------
# Level A: Core research questions
# Highest priority
# ------------------------------------------------------------

CORE_TI_KEYWORDS = [
    "transitive inference",
    "transitive reasoning",
    "transitive relational",
    "relational inference",
    "relational reasoning",
    "inferential reasoning",
    "transitive",
]


CORE_RELATIONAL_KEYWORDS = [
    "relational memory",
    "relational representation",
    "relational structure",
    "relational map",
    "relational knowledge",
    "hierarchical representation",
    "hierarchical structure",
    "schema-based inference",
    "schema-based reasoning",
    "schema learning",
    "structural learning",
    "structure learning",
    "relational learning",
]


# ------------------------------------------------------------
# Level B: Circuit mechanisms
# ------------------------------------------------------------

CIRCUIT_KEYWORDS = [
    "hippocampal-prefrontal",
    "hippocampus-prefrontal",
    "hippocampal prefrontal",
    "hippocampus prefrontal",

    "hippocampal-acc",
    "hippocampus-acc",
    "hippocampal acc",
    "hippocampus acc",

    "hippocampal-mpfc",
    "hippocampus-mpfc",
    "hippocampal mpfc",
    "hippocampus mpfc",

    "hippocampal-prelimbic",
    "hippocampus-prelimbic",
    "hippocampal prelimbic",
    "hippocampus prelimbic",

    "ventral ca1",
    "vca1",
    "ventral hippocampus",
    "ventral hippocamp",

    "prelimbic cortex",
    "prelimbic",
    "mPFC",
    "prefrontal cortex",

    "anterior cingulate cortex",
    "anterior cingulate",
    "ACC",
    "Cg1",
]


# ------------------------------------------------------------
# Level B/C: Neural mechanisms relevant to electrophysiology
# ------------------------------------------------------------

MECHANISM_KEYWORDS = [
    "electrophysiology",
    "in vivo electrophysiology",
    "local field potential",
    "LFP",
    "neural dynamics",
    "population dynamics",
    "ensemble activity",
    "population activity",

    "theta",
    "gamma",
    "theta-gamma",
    "phase-amplitude coupling",
    "PAC",
    "cross-frequency coupling",
    "spike-field coupling",

    "coherence",
    "synchrony",
    "synchronization",
    "functional connectivity",
    "directed connectivity",

    "granger causality",
    "directional connectivity",

    "sharp-wave ripple",
    "sharp wave ripple",
    "sharp-wave ripples",
    "sharp wave ripples",
    "ripple oscillation",
    "ripple oscillations",
    "ripples",

    "optogenetic",
    "optogenetics",
    "projection-specific",
    "circuit manipulation",
]


# ------------------------------------------------------------
# Level C: General cognitive/memory relevance
# ------------------------------------------------------------

GENERAL_MEMORY_KEYWORDS = [
    "memory",
    "learning",
    "memory retrieval",
    "memory recall",
    "decision making",
    "decision-making",
    "cognitive control",
    "cognitive flexibility",
    "schema",
    "retrieval",
    "recall",
]


# ------------------------------------------------------------
# Peripheral topics
# These should reduce relevance unless there is a strong
# transferable mechanism.
# ------------------------------------------------------------

LOW_PRIORITY_KEYWORDS = [
    "depression",
    "major depressive",
    "anxiety",
    "stress",
    "trauma",
    "post-traumatic",
    "fear conditioning",
    "fear extinction",
    "obesity",
    "epilepsy",
    "seizure",
    "addiction",
    "pain",
    "aging",
    "dementia",
    "alzheimer",
    "parkinson",
]


# ============================================================
# Journal tiers
# ============================================================

JOURNAL_TIERS = {

    # -------------------------
    # Tier 1
    # -------------------------

    "nature": 1,
    "science": 1,
    "cell": 1,

    "nature neuroscience": 1,
    "nature human behaviour": 1,
    "nature communications": 1,
    "nature reviews neuroscience": 1,
    "nature reviews. neuroscience": 1,

    "neuron": 1,
    "current biology": 1,
    "pnas": 1,
    "proceedings of the national academy of sciences": 1,
    "science advances": 1,

    "psychological science": 1,
    "trends in cognitive sciences": 1,
    "nature reviews psychology": 1,

    "neuroscience and biobehavioral reviews": 1,
    "annual review of psychology": 1,
    "annual review of neuroscience": 1,


    # -------------------------
    # Tier 2
    # -------------------------

    "journal of neuroscience": 2,
    "cerebral cortex": 2,
    "hippocampus": 2,
    "learning & memory": 2,
    "learning and memory": 2,

    "neurobiology of learning and memory": 2,
    "behavioral neuroscience": 2,
    "neuropsychologia": 2,

    "journal of cognitive neuroscience": 2,
    "cognitive neuroscience": 2,

    "psychonomic bulletin & review": 2,
    "psychonomic bulletin and review": 2,

    "memory & cognition": 2,
    "memory and cognition": 2,

    "journal of experimental psychology": 2,
    "journal of experimental psychology: learning memory and cognition": 2,

    "hippocampal neuroscience": 2,


    # -------------------------
    # Tier 3
    # -------------------------

    "scientific reports": 3,
    "plos one": 3,
    "plos biology": 3,
    "frontiers in neuroscience": 3,
    "frontiers in psychology": 3,

    "behavioral sciences": 3,
    "bmc neuroscience": 3,
}


# ============================================================
# Journal score
# ============================================================

def get_journal_tier(journal):
    """
    Return journal tier.
    1 = highest
    2 = strong specialized journal
    3 = general / lower priority
    4 = unknown
    """

    journal = journal.lower().strip()

    # Exact / partial matching
    for name, tier in JOURNAL_TIERS.items():
        if name in journal:
            return tier

    return 4


def journal_score(journal):
    tier = get_journal_tier(journal)

    if tier == 1:
        return 10.0
    elif tier == 2:
        return 8.0
    elif tier == 3:
        return 5.0
    else:
        return 4.0


# ============================================================
# Recency score
# ============================================================

def recency_score(publication_year):
    current_year = datetime.now().year

    try:
        year = int(publication_year)
    except Exception:
        return 4.0

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
# Scientific relevance score
# ============================================================

def count_keywords(text, keywords):
    text = text.lower()

    return sum(
        1
        for keyword in keywords
        if keyword.lower() in text
    )


def local_keyword_score(text):
    """
    Hierarchical scientific relevance scoring.

    Level A:
        Direct TI / relational inference

    Level B:
        Relational representation / hierarchy
        + hippocampal-prefrontal / ACC circuitry

    Level C:
        Electrophysiology / neural dynamics /
        general memory

    Peripheral clinical topics are penalized.
    """

    text = text.lower()

    # -----------------------------------------
    # Count keyword groups
    # -----------------------------------------

    ti_score = count_keywords(
        text,
        CORE_TI_KEYWORDS
    )

    relational_score = count_keywords(
        text,
        CORE_RELATIONAL_KEYWORDS
    )

    circuit_score = count_keywords(
        text,
        CIRCUIT_KEYWORDS
    )

    mechanism_score = count_keywords(
        text,
        MECHANISM_KEYWORDS
    )

    memory_score = count_keywords(
        text,
        GENERAL_MEMORY_KEYWORDS
    )

    low_priority_score = count_keywords(
        text,
        LOW_PRIORITY_KEYWORDS
    )


    # -----------------------------------------
    # Base hierarchical score
    # -----------------------------------------

    score = 0.0

    # Level A:
    # Direct TI / relational inference
    score += min(ti_score, 3) * 10.0

    # Relational / hierarchical representation
    score += min(relational_score, 4) * 6.0

    # Relevant circuitry
    score += min(circuit_score, 5) * 3.0

    # Neural mechanisms
    score += min(mechanism_score, 5) * 2.0

    # General memory/cognition
    score += min(memory_score, 3) * 1.0


    # -----------------------------------------
    # Penalize peripheral topics
    # -----------------------------------------

    score -= min(low_priority_score, 3) * 4.0


    # -----------------------------------------
    # Combination bonuses
    # -----------------------------------------

    has_ti = ti_score > 0
    has_relational = relational_score > 0
    has_circuit = circuit_score > 0
    has_mechanism = mechanism_score > 0


    # Direct TI + relevant circuit
    if has_ti and has_circuit:
        score += 15.0


    # Relational representation + relevant circuit
    if has_relational and has_circuit:
        score += 10.0


    # TI / relational cognition + neural mechanism
    if (has_ti or has_relational) and has_mechanism:
        score += 8.0


    # General hippocampus/PFC papers
    # receive no additional bonus.
    #
    # This is intentional:
    # hippocampus + PFC alone should not outrank
    # direct relational/TI studies.


    return max(score, 0.0)


# ============================================================
# PubMed search
# ============================================================

def search_pubmed():
    """
    Search PubMed for papers published in the last SEARCH_DAYS.
    """

    end_date = datetime.now()
    start_date = end_date - timedelta(days=SEARCH_DAYS)

    start_str = start_date.strftime("%Y/%m/%d")
    end_str = end_date.strftime("%Y/%m/%d")

    query = f"""
    (
        {PUBMED_QUERY}
    )
    AND
    ("{start_str}"[Date - Publication] : "{end_str}"[Date - Publication])
    """

    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"

    params = {
        "db": "pubmed",
        "term": query,
        "retmode": "json",
        "retmax": MAX_RESULTS,
        "sort": "pub date",
    }

    response = requests.get(
        url,
        params=params,
        timeout=30
    )

    response.raise_for_status()

    data = response.json()

    return data["esearchresult"]["idlist"]


# ============================================================
# Fetch PubMed records
# ============================================================

def fetch_pubmed_records(pmids):
    if not pmids:
        return []

    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"

    params = {
        "db": "pubmed",
        "id": ",".join(pmids),
        "retmode": "json",
    }

    response = requests.get(
        url,
        params=params,
        timeout=30
    )

    response.raise_for_status()

    data = response.json()

    results = []

    for pmid in pmids:

        item = data["result"].get(pmid)

        if not item:
            continue

        title = item.get("title", "").strip()
        journal = item.get("fulljournalname", "").strip()

        pubdate = item.get("pubdate", "")

        # Extract publication year
        match = re.search(r"(19|20)\d{2}", pubdate)

        if match:
            publication_year = int(match.group(0))
        else:
            publication_year = datetime.now().year

        results.append({
            "pmid": pmid,
            "title": title,
            "journal": journal,
            "publication_year": publication_year,
            "pubdate": pubdate,
        })

    return results


# ============================================================
# Fetch abstracts
# ============================================================

def fetch_abstracts(pmids):
    if not pmids:
        return {}

    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

    params = {
        "db": "pubmed",
        "id": ",".join(pmids),
        "retmode": "xml",
    }

    response = requests.get(
        url,
        params=params,
        timeout=30
    )

    response.raise_for_status()

    xml = response.text

    abstracts = {}

    # Split into article blocks
    articles = re.findall(
        r"<PubmedArticle>(.*?)</PubmedArticle>",
        xml,
        flags=re.DOTALL
    )

    for article in articles:

        pmid_match = re.search(
            r"<PMID[^>]*>(.*?)</PMID>",
            article
        )

        if not pmid_match:
            continue

        pmid = pmid_match.group(1)

        abstract_parts = re.findall(
            r"<AbstractText[^>]*>(.*?)</AbstractText>",
            article,
            flags=re.DOTALL
        )

        abstract = " ".join(
            re.sub(r"<[^>]+>", " ", part)
            for part in abstract_parts
        )

        abstract = re.sub(
            r"\s+",
            " ",
            abstract
        ).strip()

        abstracts[pmid] = abstract

    return abstracts


# ============================================================
# Prepare papers
# ============================================================

def prepare_papers(records, abstracts):

    papers = []

    for record in records:

        pmid = record["pmid"]

        abstract = abstracts.get(
            pmid,
            ""
        )

        searchable_text = (
            record["title"]
            + " "
            + abstract
            + " "
            + record["journal"]
        )

        local_score = local_keyword_score(
            searchable_text
        )

        recency = recency_score(
            record["publication_year"]
        )

        journal = journal_score(
            record["journal"]
        )

        preliminary_score = (
            local_score * RELEVANCE_WEIGHT
            + recency * RECENCY_WEIGHT
            + journal * JOURNAL_WEIGHT
        )

        papers.append({
            "pmid": pmid,
            "title": record["title"],
            "journal": record["journal"],
            "publication_year": record["publication_year"],
            "abstract": abstract,
            "local_score": local_score,
            "recency_score": recency,
            "journal_score": journal,
            "journal_tier": get_journal_tier(
                record["journal"]
            ),
            "preliminary_score": preliminary_score,
        })

    # Sort by local preliminary score
    papers.sort(
        key=lambda x: x["preliminary_score"],
        reverse=True
    )

    return papers[:MAX_AI_PAPERS]


# ============================================================
# DeepSeek / OpenAI-compatible client
# ============================================================

def get_ai_client():

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
        base_url=api_base
    )


# ============================================================
# AI relevance evaluation
# ============================================================

def evaluate_with_ai(papers):

    client = get_ai_client()

    paper_text = []

    for i, paper in enumerate(papers):

        paper_text.append(
            f"""
INDEX: {i}

Title:
{paper['title']}

Journal:
{paper['journal']}

Publication year:
{paper['publication_year']}

Abstract:
{paper['abstract']}
"""
        )

    combined_text = "\n".join(
        paper_text
    )


    system_prompt = """
You are a neuroscience literature screening assistant.

Evaluate each paper for relevance to the user's research program.

The user's primary research interests are:

1. Transitive inference
2. Relational inference and relational reasoning
3. Relational memory and relational representation
4. Hierarchical and schema-based representation
5. Hippocampal-prefrontal / hippocampal-ACC circuitry
6. Ventral CA1 and prelimbic cortex
7. Electrophysiological neural dynamics
8. Oscillatory coordination, theta-gamma coupling,
   ripples, population dynamics, and directed connectivity
9. Computational mechanisms underlying inference,
   especially value-based versus relational accounts

Use this relevance hierarchy.

VERY HIGH RELEVANCE:
- Direct transitive inference studies
- Direct relational inference/reasoning studies
- Studies directly testing hippocampal-prefrontal mechanisms
  of transitive or relational inference
- Rodent studies directly relevant to transitive inference
- Studies providing a directly useful experimental paradigm
  for rodent relational inference

HIGH RELEVANCE:
- Relational memory
- Relational representation
- Hierarchical representation
- Relational maps
- Schema-based inference
- Structural learning
- Hippocampal-prefrontal or hippocampal-ACC mechanisms
  closely related to memory, inference, learning, or decision making
- Neural dynamics that directly inform relational computation

MODERATE RELEVANCE:
- General hippocampal-prefrontal electrophysiology
- Hippocampal-prefrontal oscillatory coordination
- Neural population dynamics
- Theta/gamma/PAC/coherence/ripple mechanisms
- General memory or cognitive-control mechanisms

LOW RELEVANCE:
- Studies involving hippocampus or PFC only incidentally
- General motivation or reward studies without relational relevance
- Clinical or disease studies without a transferable mechanism

VERY LOW RELEVANCE:
- Depression
- Anxiety
- Stress
- Trauma
- Fear conditioning/extinction
- Obesity
- Epilepsy
- Pain
- Aging
- Dementia
- Other clinical topics

IMPORTANT:

Scientific relevance is much more important than journal prestige.

Do NOT give a high score merely because a paper is published
in Nature, Science, Neuron, PNAS, or another prestigious journal.

A highly relevant paper in a specialized journal should rank
above a prestigious but weakly relevant paper.

A paper involving hippocampus + PFC alone is NOT automatically
highly relevant.

Direct transitive inference or relational inference should
normally outrank general hippocampal-prefrontal studies.

Relational representation combined with hippocampal-prefrontal
circuit mechanisms should receive a high score.

General clinical papers should usually receive low scores even
if they mention hippocampus, PFC, or ACC.

Do not automatically reject older papers.
Older classic papers can still be important if scientifically
relevant.

Return a relevance score from 0 to 10.

Also assign one category, such as:
- Transitive inference
- Relational inference
- Relational memory
- Hierarchical representation
- Hippocampal-prefrontal circuit
- Hippocampal-ACC circuit
- Neural dynamics
- Electrophysiology
- Computational neuroscience
- General memory
- Peripheral/clinical

For "why_relevant", explain specifically why the paper matters
for the user's research rather than simply repeating keywords.

For "tldr", summarize the main scientific finding in 1-2 sentences.

Return ONLY valid JSON in this format:

[
  {
    "index": 0,
    "relevance_score": 9.5,
    "category": "Transitive inference",
    "why_relevant": "...",
    "tldr": "..."
  }
]
"""


    user_prompt = f"""
Evaluate the following papers.

{combined_text}
"""


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
        temperature=0.1,
    )


    content = response.choices[0].message.content.strip()

    # Remove accidental Markdown code fences
    content = re.sub(
        r"^```json\s*",
        "",
        content
    )

    content = re.sub(
        r"\s*```$",
        "",
        content
    )

    return json.loads(content)


# ============================================================
# Combine AI score + recency + journal
# ============================================================

def combine_scores(papers, ai_results):

    ai_map = {
        item["index"]: item
        for item in ai_results
    }

    final_papers = []

    for index, paper in enumerate(papers):

        ai = ai_map.get(index)

        if not ai:
            continue

        relevance = float(
            ai.get("relevance_score", 0)
        )

        recency = paper["recency_score"]
        journal = paper["journal_score"]

        final_score = (
            relevance * RELEVANCE_WEIGHT
            + recency * RECENCY_WEIGHT
            + journal * JOURNAL_WEIGHT
        )

        paper["relevance_score"] = relevance
        paper["final_score"] = final_score

        paper["category"] = ai.get(
            "category",
            "Uncategorized"
        )

        paper["why_relevant"] = ai.get(
            "why_relevant",
            ""
        )

        paper["tldr"] = ai.get(
            "tldr",
            ""
        )

        final_papers.append(paper)

    final_papers = [
    paper
    for paper in final_papers
    if paper["relevance_score"] >= MIN_RELEVANCE_SCORE
]

final_papers.sort(
    key=lambda x: x["final_score"],
    reverse=True
)

return final_papers[:TOP_N]


# ============================================================
# Print results
# ============================================================

def print_results(papers):

    print("\n")
    print("=" * 70)
    print("TOP 15 PAPERS")
    print("=" * 70)

    for i, paper in enumerate(
        papers,
        start=1
    ):

        print(
            f"\n[{i}] "
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

        tier = paper["journal_tier"]

        print(
            f"Journal: "
            f"{paper['journal']} "
            f"(Tier {tier})"
        )

        print(
            f"Publication year: "
            f"{paper['publication_year']}"
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
            f"https://pubmed.ncbi.nlm.nih.gov/"
            f"{paper['pmid']}/"
        )


# ============================================================
# Main
# ============================================================

def main():

    print(
        "Searching PubMed..."
    )

    pmids = search_pubmed()

    print(
        f"Found {len(pmids)} PubMed records."
    )

    if not pmids:
        print(
            "No papers found."
        )
        return


    print(
        "Fetching PubMed records..."
    )

    records = fetch_pubmed_records(
        pmids
    )


    print(
        "Fetching abstracts..."
    )

    abstracts = fetch_abstracts(
        pmids
    )


    print(
        "Applying local scientific relevance filter..."
    )

    papers = prepare_papers(
        records,
        abstracts
    )

    print(
        f"Sending top {len(papers)} papers "
        f"to DeepSeek..."
    )


    ai_results = evaluate_with_ai(
        papers
    )


    final_papers = combine_scores(
        papers,
        ai_results
    )


    print_results(
        final_papers
    )


if __name__ == "__main__":
    main()
