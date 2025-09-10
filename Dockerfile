FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir pipenv

COPY data/netflix_titles_enriched_full.csv ./data/netflix_titles_enriched_full.csv
COPY ["Pipfile", "Pipfile.lock", "./"]

RUN pipenv install --deploy --ignore-pipfile --system

COPY src/modules/workflow/ ./src/modules/workflow/
COPY src/modules/search/ ./src/modules/search/

EXPOSE 5000

CMD gunicorn --bind 0.0.0.0:5000 src.modules.workflow.app:app

