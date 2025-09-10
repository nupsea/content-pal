from src.modules.workflow import ingest

from openai import OpenAI
from typing import List, Dict, Any
from string import Template


llm_client = OpenAI()

search_system = ingest.load_index()


def full_asset_search(query: str, k=10) -> List[Dict[str, Any]]:
    results = search_system.search(query, top_k=k)

    docs = []
    for res in results:
        src = res.metadata
        doc = {
            "show_id": res.id,
            "type": res.content_type,
            "title": res.title,
            "director": src.get("director"),
            "cast": src.get("cast"),
            "country": src.get("country"),
            "date_added": src.get("date_added"),
            "release_year": src.get("release_year"),
            "rating": src.get("rating"),
            "duration": src.get("duration"),
            "listed_in": src.get("listed_in"),
            "description": src.get("description"),
        }
        docs.append(doc)
    return docs

def generate_response(q):
    response = llm_client.chat.completions.create(
        model='gpt-4o-mini',
        messages=[{"role": "user", "content": q}]
    )

    json_response = response.choices[0].message.content
    return json_response


entry_template = Template("""
show_id: $show_id
type: $type
title: $title
director: $director
cast: $cast
country: $country
date_added: $date_added
release_year: $release_year
rating: $rating
duration: $duration
listed_in: $listed_in
description: $description
""")

prompt_template = Template("""
You are a streaming-catalog assistant.

Return ONE JSON object matching this SCHEMA exactly (no extra keys, no prose):

SCHEMA: {
  "catalog_recommendations": [
    "recommendation 1",
    "recommendation 2",
    "recommendation 3",
    "recommendation 4",
    "recommendation 5"
  ]
}

HARD RULES
- CONTEXT is pre-ranked (earlier = more relevant). Build "catalog_recommendations" ONLY from CONTEXT but feel free to change order.
- Scan CONTEXT top-down:
  1) Add items that plausibly match QUERY.
  2) If fewer than MIN_CATALOG and CONTEXT still has items, keep taking the next items (even weak matches)
     until you reach MIN_CATALOG or run out of CONTEXT.
- If CONTEXT has >= MIN_CATALOG items total, you MUST return at least MIN_CATALOG in "catalog_recommendations".
- Copy fields exactly from CONTEXT; for "cast", split the comma-separated string and trim; drop empties.
- Deduplicate by (title, release_year); keep the earlier one.
- Final counts:
  len(catalog_recommendations) >= min(MIN_CATALOG, number_of_items_in_CONTEXT)
- Recommendations must of the format "Title (Release Year): Small Description snippet"


INPUTS
QUERY: $query
MIN_CATALOG: $min_catalog

CONTEXT (ranked):
$context
""".strip())



def build_prompt(query, search_results, allow_external=True, min_catalog=5):
    context = ""
    for doc in search_results:
        context += entry_template.substitute(**doc) + "\n\n"
    return prompt_template.substitute(
        query=query,
        context=context,
        allow_external=str(allow_external).lower(), 
        min_catalog=min_catalog,
    )

def llm(prompt):
    response = llm_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content

def rag(query):
    search_results = full_asset_search(query, k=10)
    if not search_results:
        print("WARN: No relevant results found.")

    prompt = build_prompt(query, search_results)
    response = llm(prompt)
    return response
