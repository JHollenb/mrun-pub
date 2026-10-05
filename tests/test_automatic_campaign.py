from __future__ import annotations

import numpy as np
import torch

from mrun.engine.paged import PagedEngine


class _AutomaticCampaignEngine:
    def __init__(self) -> None:
        self.calls: list[tuple[tuple[tuple[int, ...], ...], tuple[int, ...]]] = []

    def encode(self, texts: list[str], *, add_special_tokens: bool) -> list[np.ndarray]:
        assert add_special_tokens is False
        table = {
            "shared": (10, 11, 12),
            "other": (20, 21),
            "A": (1,),
            "B": (2,),
            "C": (3,),
            "D": (4,),
        }
        return [np.asarray(table[text], dtype=np.int64) for text in texts]

    def selected_last_logits_batch(
        self,
        prompts: list[np.ndarray],
        union: tuple[int, ...],
    ) -> torch.Tensor:
        self.calls.append(
            (tuple(tuple(int(value) for value in row) for row in prompts), tuple(union))
        )
        return torch.tensor(
            [[float(row[0] + token) for token in union] for row in prompts],
            dtype=torch.float32,
        )


def test_ordinary_forced_choice_subset_automatically_quotients_exact_prompts_and_rows() -> None:
    engine = _AutomaticCampaignEngine()
    probes = [
        {"probe_id": "q1", "prompt": "shared", "answers": ["A", "B"]},
        {"probe_id": "q2", "prompt": "shared", "answers": ["B", "C"]},
        {"probe_id": "q3", "prompt": "other", "answers": ["A", "D"]},
    ]
    result = PagedEngine.score_forced_choice_argmax_subset(engine, probes)  # type: ignore[arg-type]

    assert engine.calls == [(((10, 11, 12), (20, 21)), (1, 2, 3, 4))]
    assert [row["probe_id"] for row in result["rows"]] == ["q1", "q2", "q3"]
    assert [row["margin"] for row in result["rows"]] == [-1.0, -1.0, -3.0]
    assert not any(row["correct_ranked_first"] for row in result["rows"])
    assert result["automatic_campaign"] == {
        "eligible": 3,
        "fallback": 0,
        "logical_prompt_evaluations": 3,
        "physical_prompt_evaluations": 2,
        "candidate_references": 6,
        "candidate_union_rows": 4,
    }
