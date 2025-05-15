# solver.py
import tempfile, os
from clingo import Control
from config import DOMAIN_RULES_FILE
from init_db import get_facts_for_document

def check_consistency(document_id: int) -> str:
    ctl = Control()
    ctl.load(DOMAIN_RULES_FILE)
    ctl.ground([("base", [])])

    # fetch extracted facts from DB
    facts = get_facts_for_document(document_id)
    # write them to a temp file
    fd, path = tempfile.mkstemp(suffix=".lp")
    with os.fdopen(fd, "w") as f:
        f.write("\n".join(facts))

    ctl.add("doc", [], f'#include "{path}".')
    ctl.ground([("doc", [])])

    result = ctl.solve()
    os.remove(path)
    return "CONSISTENT" if result.satisfiable else "UNSATISFIABLE"
