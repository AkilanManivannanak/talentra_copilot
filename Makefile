.PHONY: setup test lint eval eval-ablation benchmark smoke run-api run-ui docker labels

setup:
	python3 -m venv .venv && . .venv/bin/activate && \
	pip install -r requirements.txt && python -m spacy download en_core_web_sm

setup-ml:  ## optional: pretrained bi-encoder + cross-encoder reranker
	. .venv/bin/activate && pip install -r requirements-ml.txt

lint:
	ruff check app eval scripts tests

test:
	pytest -q --cov=app --cov-report=term-missing

labels:
	python eval/build_labels.py

eval:
	python eval/harness.py --mode hybrid --k 5

eval-ablation:
	python eval/harness.py --ablation --k 5

benchmark:
	python scripts/benchmark.py

smoke: lint test eval benchmark

run-api:
	uvicorn app.main:app --reload

run-ui:
	streamlit run frontend/streamlit_app.py

docker:
	docker build -t talentra-copilot . && docker run --rm -p 8000:8000 talentra-copilot
