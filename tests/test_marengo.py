import os
import pytest

from lighthouse.models import MarengoPredictor


MOMENT_NUM = 10
EMBEDDING_DIM = 512


def test_predict_ranks_clips_by_cosine_similarity():
    """No-network unit test: predict() must rank clips by query similarity."""
    predictor = MarengoPredictor.__new__(MarengoPredictor)
    predictor._moment_num = MOMENT_NUM

    # clip 0 points the same way as the query, clip 1 is orthogonal.
    inputs = {
        'segments': [
            [0.0, 2.0, 1.0, 0.0],
            [2.0, 4.0, 0.0, 1.0],
        ],
    }
    predictor._encode_text = lambda query: [1.0, 0.0]  # type: ignore[method-assign]

    prediction = predictor.predict('a query', inputs)
    windows = prediction['pred_relevant_windows']

    assert len(windows) == 2
    assert windows[0][:2] == [0.0, 2.0], 'the aligned clip should rank first'
    assert windows[0][2] == 1.0
    assert windows[1][2] == 0.0
    assert windows[0][2] >= windows[1][2], 'windows must be sorted by descending score'


def test_predict_without_segments_returns_none():
    predictor = MarengoPredictor.__new__(MarengoPredictor)
    predictor._moment_num = MOMENT_NUM
    assert predictor.predict('a query', {'segments': []}) is None


@pytest.mark.skipif(
    'TWELVELABS_API_KEY' not in os.environ,
    reason='TWELVELABS_API_KEY not set; skipping live Marengo API test.')
def test_marengo_text_embedding_is_512_dim():
    """Live smoke test: a Marengo text embedding is a 512-dim vector."""
    predictor = MarengoPredictor(clip_length=2.0)
    embedding = predictor._encode_text('a person walking a dog')
    assert len(embedding) == EMBEDDING_DIM
