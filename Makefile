.PHONY: train-baseline build-rag app api test lint

train-baseline:
	PYTHONPATH=src python scripts/train_baseline_demo.py

build-rag:
	PYTHONPATH=src python -m complaint_ai.rag.build_index --policy_dir knowledge_base/policies --index_path models/policy_faiss.index --metadata_path models/policy_metadata.json

app:
	PYTHONPATH=src streamlit run app/streamlit_app.py

api:
	PYTHONPATH=src uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

test:
	PYTHONPATH=src pytest -q

lint:
	PYTHONPATH=src python -m compileall src app api scripts

powerbi-export:
	PYTHONPATH=src python scripts/export_powerbi_tables.py
