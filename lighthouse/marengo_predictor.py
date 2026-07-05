import os

from typing import Dict, List, Optional

"""
Copyright $today.year LY Corporation

LY Corporation licenses this file to you under the Apache License,
version 2.0 (the "License"); you may not use this file except in compliance
with the License. You may obtain a copy of the License at:

  https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
License for the specific language governing permissions and limitations
under the License.
"""


class TwelveLabsPredictor:
    """
    Zero-shot video moment retrieval backed by TwelveLabs embeddings.

    Unlike the other predictors in :mod:`lighthouse.models`, this one needs no
    local checkpoint or feature files. It segments the video on the server side,
    embeds each clip and the text query into the same 512-dim embedding space,
    and ranks clips by cosine similarity. The output mirrors the rest of the
    library:

        {"pred_relevant_windows": [[start, end, score], ...]}

    The embedding model is configurable via the ``model_name`` argument and
    defaults to Marengo (``marengo3.0``), so the class can support future
    TwelveLabs embedding models without code changes.

    This is an opt-in addition; existing predictors and their behaviour are
    unchanged. It depends on the official ``twelvelabs`` SDK, which is only
    imported when this class is instantiated, so users who do not use it pay no
    import cost.

    A free API key with a generous free tier is available at https://twelvelabs.io.
    """

    DEFAULT_MODEL_NAME: str = 'marengo3.0'
    EMBEDDING_DIM: int = 512

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = DEFAULT_MODEL_NAME,
        clip_length: float = 2.0,
        moment_num: int = 10) -> None:

        try:
            from twelvelabs import TwelveLabs
        except ImportError as e:
            raise ImportError(
                'TwelveLabsPredictor requires the twelvelabs SDK. '
                "Install it with: pip install 'lighthouse[twelvelabs]' "
                'or pip install twelvelabs.') from e

        resolved_key = api_key or os.environ.get('TWELVELABS_API_KEY')
        if not resolved_key:
            raise ValueError(
                'A TwelveLabs API key is required. Pass api_key=... or set the '
                'TWELVELABS_API_KEY environment variable. Get a free key at '
                'https://twelvelabs.io.')

        self._client = TwelveLabs(api_key=resolved_key)
        self.model_name: str = model_name
        self._clip_length: float = clip_length
        self._moment_num: int = moment_num

    @staticmethod
    def _cosine_similarity(
        query: List[float],
        clip: List[float]) -> float:
        dot = sum(q * c for q, c in zip(query, clip))
        query_norm = sum(q * q for q in query) ** 0.5
        clip_norm = sum(c * c for c in clip) ** 0.5
        if query_norm == 0.0 or clip_norm == 0.0:
            return 0.0
        return dot / (query_norm * clip_norm)

    def encode_video(
        self,
        video_path: Optional[str] = None,
        video_url: Optional[str] = None) -> Dict[str, List[List[float]]]:
        """
        Embed every clip of a video with TwelveLabs.

        Provide either a local ``video_path`` or a publicly reachable
        ``video_url``. Returns a dict with one ``segments`` entry, each being
        ``[start_offset_sec, end_offset_sec, *embedding]``, ready to pass to
        :meth:`predict`.
        """
        if (video_path is None) == (video_url is None):
            raise ValueError('Pass exactly one of video_path or video_url.')

        if video_path is not None:
            with open(video_path, 'rb') as f:
                task = self._client.embed.tasks.create(
                    model_name=self.model_name,
                    video_file=f,
                    video_clip_length=self._clip_length,
                    video_embedding_scope=['clip'])
        else:
            task = self._client.embed.tasks.create(
                model_name=self.model_name,
                video_url=video_url,
                video_clip_length=self._clip_length,
                video_embedding_scope=['clip'])

        if task.id is None:
            raise RuntimeError('The embedding task creation returned no task id.')
        task_id: str = task.id

        # wait_for_done lives on the SDK's runtime wrapper class, which mypy
        # cannot see through the statically-typed base client.
        self._client.embed.tasks.wait_for_done(task_id=task_id)  # type: ignore[attr-defined]
        result = self._client.embed.tasks.retrieve(task_id=task_id)

        if result.video_embedding is None or result.video_embedding.segments is None:
            raise RuntimeError(f'The embedding task {task_id} returned no segments.')

        segments: List[List[float]] = []
        for segment in result.video_embedding.segments:
            if segment.float_ is None:
                continue
            start = float(segment.start_offset_sec or 0.0)
            end = float(segment.end_offset_sec or 0.0)
            segments.append([start, end] + list(segment.float_))
        return {'segments': segments}

    def _encode_text(
        self,
        query: str) -> List[float]:
        response = self._client.embed.create(model_name=self.model_name, text=query)
        if response.text_embedding is None or not response.text_embedding.segments:
            raise RuntimeError('TwelveLabs returned no text embedding for the query.')
        embedding = response.text_embedding.segments[0].float_
        if embedding is None:
            raise RuntimeError('TwelveLabs returned an empty text embedding for the query.')
        return list(embedding)

    def predict(
        self,
        query: str,
        inputs: Dict[str, List[List[float]]]) -> Optional[Dict[str, List[List[float]]]]:
        """
        Rank the encoded video clips by similarity to ``query``.

        Returns ``{"pred_relevant_windows": [[start, end, score], ...]}`` sorted
        by descending score, capped at ``moment_num`` windows, matching the
        prediction format of the other Lighthouse predictors.
        """
        segments = inputs.get('segments')
        if not segments:
            print('Error: No encoded clips found. Did you forget to call encode_video()?')
            return None

        query_embedding = self._encode_text(query)

        ranked: List[List[float]] = []
        for segment in segments:
            start, end = segment[0], segment[1]
            clip_embedding = segment[2:]
            score = self._cosine_similarity(query_embedding, clip_embedding)
            ranked.append([float(f'{start:.4f}'), float(f'{end:.4f}'), float(f'{score:.4f}')])

        ranked.sort(key=lambda window: window[2], reverse=True)
        return {'pred_relevant_windows': ranked[:self._moment_num]}
