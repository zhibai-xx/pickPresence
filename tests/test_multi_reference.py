import math

from pickpresence.identity import FaceMatcher, ReferenceEmbedding


def test_face_matcher_multi_max_uses_best_reference():
    ref_a = ReferenceEmbedding(name="Front", vector=[1.0, 0.0, 0.0], ref_id="front")
    ref_b = ReferenceEmbedding(name="Side", vector=[0.0, 1.0, 0.0], ref_id="side")
    matcher = FaceMatcher([ref_a, ref_b], threshold=0.5, agg="max")
    details = matcher.match_details([0.0, 1.0, 0.0])
    assert details.best_ref_id == "side"
    assert math.isclose(details.score, 1.0, rel_tol=1e-6)


def test_face_matcher_topk_avg_aggregates():
    ref_a = ReferenceEmbedding(name="A", vector=[1.0, 0.0, 0.0], ref_id="a")
    ref_b = ReferenceEmbedding(name="B", vector=[0.0, 1.0, 0.0], ref_id="b")
    ref_c = ReferenceEmbedding(name="C", vector=[0.0, 0.0, 1.0], ref_id="c")
    matcher = FaceMatcher([ref_a, ref_b, ref_c], threshold=0.1, agg="topk_avg", topk=2)
    details = matcher.match_details([1.0, 0.5, 0.0])
    assert details.topk_avg is not None
    assert details.score == details.topk_avg
    assert 0.6 < details.score < 0.8


def test_face_matcher_single_reference_stays_consistent():
    ref = ReferenceEmbedding(name="Solo", vector=[1.0, 0.0, 0.0], ref_id="solo")
    matcher = FaceMatcher(ref, threshold=0.5, agg="topk_avg", topk=3)
    details = matcher.match_details([1.0, 0.0, 0.0])
    assert details.best_ref_id == "solo"
    assert math.isclose(details.score, 1.0, rel_tol=1e-6)
