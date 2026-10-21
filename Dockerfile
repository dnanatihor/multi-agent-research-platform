FROM python:3.12.4-slim
WORKDIR /app
RUN pip install --no-cache-dir poetry==2.5.1 \
    && useradd --uid 10001 --create-home app
COPY pyproject.toml poetry.lock poetry.toml README.md ./
COPY src ./src
RUN poetry config virtualenvs.create false \
    && poetry install --no-interaction --without dev
USER app
EXPOSE 8004
CMD ["uvicorn", "research_platform.main:app", "--host", "0.0.0.0", "--port", "8004"]
