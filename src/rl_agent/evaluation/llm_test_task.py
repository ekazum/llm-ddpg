"""``LLMTestTask`` — scores the best dev snapshot on test, once, after training and dev scoring have ended."""

import logging
from pathlib import Path

from rl_core.eval_store import EvalStore
from rl_core.test_task import TestTask

from src.rl_agent.evaluation.llm_score_task import LLMScoreTask
from src.rl_agent.token_policy_model import TokenPolicyModel

logger = logging.getLogger(__name__)


class LLMTestTask(TestTask):
    """Loads the dev champion's actor directory into the eval policy and scores it on test through ``LLMScoreTask``."""

    def __init__(self, policy: TokenPolicyModel, score: LLMScoreTask, store: EvalStore) -> None:
        """Bind the eval policy, the test score task and the store that names the champion."""
        self._policy = policy
        self._score = score
        self._store = store

    def run(self) -> dict:
        """The champion's test metrics, with its update; they are written to ``eval.db``."""
        champion = self._store.champion()
        if champion is None:
            raise RuntimeError("LLMTestTask: no dev row, so no snapshot to test")
        update, checkpoint = champion
        logger.info("LLMTestTask: best dev snapshot at update %d, %s", update, checkpoint)
        self._policy.load_weights(Path(checkpoint))
        metrics = self._score.run("test")
        self._store.add_test({"champion_update": update})
        return {**metrics, "champion_update": update}
