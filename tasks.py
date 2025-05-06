# tasks.py
from celery_app import app
from init_db import init_db, store_document, store_facts, store_result
from extract import extract_facts
from solver import check_consistency

# ensure DB tables exist when worker starts
@app.on_after_configure.connect
def init_db_tables(sender, **kwargs):
    init_db()

@app.task
def process_document(doc_path: str) -> dict:
    # 1. record document
    doc_id = store_document(doc_path)
    # 2. extract & store facts
    facts = extract_facts(doc_path)
    store_facts(doc_id, facts)
    # 3. consistency check
    status = check_consistency(doc_id)
    store_result(doc_id, status)
    return {"doc_id": doc_id, "status": status}
