from benchmark.compare_hard import aggregate, doc_counts, doc_metrics, markdown_table, select_hard


def rec(idx, tp, fp, fn, gold=("a",), ocr_s=None):
    return {"idx": idx, "gold": {k: [1, "g"] for k in gold}, "tp": tp, "fp": fp, "fn": fn, "ocr_s": ocr_s}


def test_doc_counts_f1():
    c = doc_counts(rec(0, ["a", "b"], ["c"], ["d"]))
    assert (c.tp, c.fp, c.fn) == (2, 1, 1)
    assert round(c.f1, 4) == round(2 / 3, 4)


def test_select_lowest_f1_ties_by_idx_and_skips_no_gold():
    records = [
        rec(5, [], ["x"], ["a"]),  # F1 0
        rec(2, [], [], ["a"]),  # F1 0
        rec(7, ["a"], [], []),  # F1 1
        rec(1, [], [], [], gold=()),  # no gold: skipped
        rec(3, ["a"], ["x"], ["b"]),  # F1 0.5
    ]
    assert [r["idx"] for r in select_hard(records, 3)] == [2, 5, 3]
    assert len(select_hard(records, 10)) == 4


def test_aggregate_and_table():
    rows = [
        {"idx": 1, "base": doc_metrics(doc_counts(rec(1, [], [], ["a"])), 1.0),
         "candidate": doc_metrics(doc_counts(rec(1, ["a"], [], [])), None)},
        {"idx": 2, "base": doc_metrics(doc_counts(rec(2, [], ["x"], ["a"])), 3.0),
         "candidate": doc_metrics(doc_counts(rec(2, [], [], ["a"])), 2.0)},
    ]
    agg = aggregate(rows)
    assert agg["documents"] == 2
    assert agg["base"]["mean_f1"] == 0.0 and agg["candidate"]["mean_f1"] == 0.5
    assert agg["base"]["mean_ocr_s"] == 2.0 and agg["candidate"]["mean_ocr_s"] == 2.0
    assert "| 1 |" in markdown_table(rows, agg, "s", "x")


def test_aggregate_empty():
    assert aggregate([])["base"]["mean_f1"] is None
