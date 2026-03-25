setup:
	python3 -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt

test:
	pytest -q

run-api:
	uvicorn app.main:app --reload

run-ui:
	streamlit run frontend/streamlit_app.py

benchmark:
	python scripts/benchmark.py

smoke:
	pytest -q && python scripts/benchmark.py --assert
