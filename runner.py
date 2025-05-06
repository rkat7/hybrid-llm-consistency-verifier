# runner.py
import glob
from tasks import process_document

if __name__ == "__main__":
    docs = glob.glob("corpus/*.txt")
    for path in docs:
        process_document.delay(path)
    print(f"Enqueued {len(docs)} documents.")
