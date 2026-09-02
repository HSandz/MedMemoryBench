import sys
import os
os.environ["OPENAI_API_KEY"] = "sk-dummy"
sys.path.append("/Users/hwidg/Desktop/KTLab/MedMemoryBench-main")

from methods.smart_mem0.router import DeterministicRouter
from methods.smart_mem0.p1a_execution import _prepare_p1a_query

def test_whitelist_hard():
    r = DeterministicRouter()
    # Visibility options
    assert r.route_query("Do I need to adjust my medication now?\nA. Yes\nB. No").route == "HARD"
    assert r.route_query("Could this be related to my diet?").route == "HARD"
    assert r.route_query("Does this mean it's worsening?").route == "HARD"
    assert r.route_query("Do I need to keep a close eye on it?").route == "HARD"
    assert r.route_query("Compare A and B").route == "HARD"

def test_subject_resolution():
    r = DeterministicRouter(subject_postings={"primary_user": set()})
    assert r._resolve_subject("What is my latest hba1c?") == "primary_user"
    assert r._resolve_subject("What is the latest status?") == "primary_user" # 1 subject in ledger
    
    r2 = DeterministicRouter(subject_postings={"primary_user": set(), "third_party:mother": set()})
    assert r2._resolve_subject("What is the latest status?") == "HARD" # ambiguous
    assert r2._resolve_subject("What is my mother's disease?") == "third_party:mother"

def test_temporal_anchors():
    r = DeterministicRouter()
    d = r.route_query("What was my hba1c on April 3?")
    assert d.route == "TEMPORAL"
    assert d.temporal_month == 4
    assert d.temporal_day == 3
    
    d2 = r.route_query("What was my hba1c in February 2024?")
    assert d2.route == "TEMPORAL"
    assert d2.temporal_year == 2024
    assert d2.temporal_month == 2

test_whitelist_hard()
test_subject_resolution()
test_temporal_anchors()
print("All routing tests passed!")
