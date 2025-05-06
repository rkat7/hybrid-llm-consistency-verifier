# init_db.py
import sqlite3
from config import SQLITE_PATH

def get_conn():
    return sqlite3.connect(SQLITE_PATH)

def init_db():
    conn = get_conn()
    c = conn.cursor()
    c.execute("""CREATE TABLE IF NOT EXISTS documents (
                   id   INTEGER PRIMARY KEY,
                   path TEXT UNIQUE
                 );""")
    c.execute("""CREATE TABLE IF NOT EXISTS facts (
                   id          INTEGER PRIMARY KEY,
                   document_id INTEGER,
                   fact        TEXT,
                   FOREIGN KEY(document_id) REFERENCES documents(id)
                 );""")
    c.execute("""CREATE TABLE IF NOT EXISTS results (
                   document_id INTEGER PRIMARY KEY,
                   status      TEXT,
                   FOREIGN KEY(document_id) REFERENCES documents(id)
                 );""")
    conn.commit(); conn.close()

def store_document(path):
    conn = get_conn()
    c = conn.cursor()
    c.execute("INSERT OR IGNORE INTO documents (path) VALUES (?)", (path,))
    conn.commit()
    c.execute("SELECT id FROM documents WHERE path = ?", (path,))
    doc_id = c.fetchone()[0]
    conn.close()
    return doc_id

def store_facts(document_id, facts):
    conn = get_conn()
    c = conn.cursor()
    c.executemany(
      "INSERT INTO facts (document_id, fact) VALUES (?, ?)",
      [(document_id, f) for f in facts]
    )
    conn.commit(); conn.close()

def get_facts_for_document(document_id):
    conn = get_conn()
    c = conn.cursor()
    c.execute("SELECT fact FROM facts WHERE document_id = ?", (document_id,))
    facts = [row[0] for row in c.fetchall()]
    conn.close()
    return facts

def store_result(document_id, status):
    conn = get_conn()
    c = conn.cursor()
    c.execute(
      "INSERT OR REPLACE INTO results (document_id, status) VALUES (?, ?)",
      (document_id, status)
    )
    conn.commit(); conn.close()
