from src.modules.workflow import ingest

from openai import OpenAI
from typing import List, Dict, Any
from string import Template
from time import time

import json


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

def llm(prompt, model="gpt-4o-mini"):
    response = llm_client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    answer = response.choices[0].message.content

    token_stats = {
        "prompt_tokens": response.usage.prompt_tokens,
        "completion_tokens": response.usage.completion_tokens,
        "total_tokens": response.usage.total_tokens,
    }

    return answer, token_stats



prompt_judge_template = """
You are an expert evaluator for a Retrieval-Augmented Generation (RAG) system.
Your task is to analyze the relevance of the generated content to the user's question.
Based on the relevance of the available content, you will classify it
as "NON_RELEVANT", "PARTLY_RELEVANT", or "RELEVANT".

Here is the data for evaluation:

Question: {question}
Generated Answer: {answer_llm}

Please analyze the content and context of the generated answer in relation to the question
and provide your evaluation in parsable JSON without using code blocks:

{{
  "Relevance": "NON_RELEVANT" | "PARTLY_RELEVANT" | "RELEVANT",
  "Explanation": "[Provide a brief explanation for your evaluation]"
}}
""".strip()


def evaluate_relevance(question, answer):
    prompt = prompt_judge_template.format(question=question, answer_llm=answer)
    evaluation, tokens = llm(prompt, model="gpt-4o-mini")

    try:
        json_eval = json.loads(evaluation)
        return json_eval, tokens
    except json.JSONDecodeError:
        result = {"Relevance": "UNKNOWN", "Explanation": "Failed to parse evaluation"}
        return result, tokens


def calculate_openai_cost(model, tokens):
    openai_cost = 0

    if model == "gpt-4o-mini":
        openai_cost = (
            tokens["prompt_tokens"] * 0.00015 + tokens["completion_tokens"] * 0.0006
        ) / 1000
    else:
        print("Model not recognized. OpenAI cost calculation failed.")

    return openai_cost


def rag(query, model="gpt-4o-mini"):
    t0 = time()

    search_results = full_asset_search(query)
    prompt = build_prompt(query, search_results)
    answer, token_stats = llm(prompt, model=model)

    relevance, rel_token_stats = evaluate_relevance(query, answer)

    t1 = time()
    took = t1 - t0

    openai_cost_rag = calculate_openai_cost(model, token_stats)
    openai_cost_eval = calculate_openai_cost(model, rel_token_stats)

    openai_cost = openai_cost_rag + openai_cost_eval

    answer_data = {
        "answer": answer,
        "model_used": model,
        "response_time": took,
        "relevance": relevance.get("Relevance", "UNKNOWN"),
        "relevance_explanation": relevance.get(
            "Explanation", "Failed to parse evaluation"
        ),
        "prompt_tokens": token_stats["prompt_tokens"],
        "completion_tokens": token_stats["completion_tokens"],
        "total_tokens": token_stats["total_tokens"],
        "eval_prompt_tokens": rel_token_stats["prompt_tokens"],
        "eval_completion_tokens": rel_token_stats["completion_tokens"],
        "eval_total_tokens": rel_token_stats["total_tokens"],
        "openai_cost": openai_cost,
    }

    return answer_data