"""Unit tests for the segmentation package (labels, metrics, decoding).

These only hit the pure functions, so no model or checkpoint is needed.
Run:  python tests/test_segmentation.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "segmentation"))
from data import meeting_to_labels
from metrics import pk_windowdiff, mean_scores
from baselines import never_split
from segment_transcript import decode_boundaries


# meeting_to_labels

def test_labels_mark_topic_starts():
    meeting = {
        "meeting_transcripts": [{"content": c} for c in ["a", "b", "c", "d", "e"]],
        "topic_list": [{"relevant_text_span": [["0", "2"]]},
                       {"relevant_text_span": [["3", "4"]]}],
    }
    texts, labels = meeting_to_labels(meeting)
    assert texts == ["a", "b", "c", "d", "e"]
    assert labels == [1, 0, 0, 1, 0]


def test_first_turn_is_always_a_boundary():
    meeting = {"meeting_transcripts": [{"content": "x"}, {"content": "y"}], "topic_list": []}
    _, labels = meeting_to_labels(meeting)
    assert labels == [1, 0]


def test_out_of_range_spans_are_ignored():
    meeting = {"meeting_transcripts": [{"content": "x"}, {"content": "y"}],
               "topic_list": [{"relevant_text_span": [["5", "9"]]}]}
    _, labels = meeting_to_labels(meeting)
    assert labels == [1, 0]


# metrics (Pk / WindowDiff)

def test_perfect_prediction_scores_zero():
    labels = [1, 0, 0, 1, 0, 0, 1, 0, 0]
    assert pk_windowdiff(labels, labels) == (0.0, 0.0)


def test_wrong_prediction_scores_above_zero():
    ref = [1, 0, 0, 1, 0, 0, 1, 0, 0]
    hyp = [1, 0, 0, 0, 0, 0, 0, 0, 0]   # one big segment, misses the real boundaries
    pk, wd = pk_windowdiff(ref, hyp)
    assert pk > 0 and wd > 0


def test_mean_scores_averages_pairs():
    labels = [1, 0, 0, 1, 0, 0]
    assert mean_scores([(labels, labels), (labels, labels)]) == (0.0, 0.0)


# never-split baseline

def test_never_split_is_one_segment():
    assert never_split(4) == [1, 0, 0, 0]
    assert never_split(1) == [1]


# decode_boundaries

def test_threshold_decoding_picks_high_probs():
    probs = [0.1, 0.9, 0.2, 0.8, 0.1]
    assert decode_boundaries(probs, threshold=0.5) == [0, 1, 3]


def test_decode_always_starts_at_zero():
    bounds = decode_boundaries([0.5] * 20, num_sections=4)
    assert bounds[0] == 0
    assert bounds == sorted(bounds)


def test_decode_respects_min_gap():
    bounds = decode_boundaries([0.5] * 40, num_sections=8, min_gap=3)
    gaps = [b - a for a, b in zip(bounds, bounds[1:])]
    assert all(g >= 3 for g in gaps)


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS {t.__name__}")
    print(f"\n{len(tests)} tests passed.")
