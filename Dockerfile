FROM langchain/langgraph-api:3.11

COPY . /app
WORKDIR /app

RUN --mount=type=cache,target=/root/.cache/pip PYTHONDONTWRITEBYTECODE=1 pip install -c /api/constraints.txt -e .

# Add any other necessary commands 