"""LLM-as-judge answer correctness for QA evaluation.

Exact-match and token-F1 penalize answers that are correct but phrased
differently from the gold string -- a paraphrase ("Academy Award nomination for
Best Supporting Actor" vs. gold "nominated for an Academy Award for Best
Supporting Actor"), a differing granularity ("King James I" vs. "King James I of
England"), or extra surrounding context. On MuSiQue these account for a large
share of the apparent errors. This module adds a model-based judge that decides
whether a prediction actually answers the question given the gold answer(s).

The judge reuses the single configured ``GeneratorModule`` (the same
reuse-one-generator pattern as ``AnswerVerifier``). It is a supplementary metric:
it is model-based and non-deterministic, so it is reported alongside -- never as a
replacement for -- exact-match / F1, which stay comparable to published numbers.
"""

import logging
from typing import List

from agrag.constants import LOGGER_NAME

logger = logging.getLogger(LOGGER_NAME)

# Guard rails on prompt size: predictions in non-agentic mode can be long
# chain-of-thought text. Truncating keeps the judge prompt bounded without
# dropping the (usually leading or trailing) answer span in practice.
_MAX_PREDICTION_CHARS = 4000

_JUDGE_INSTRUCTION = (
    "You are grading answers to a question against the reference (gold) "
    "answer(s). Decide whether the PREDICTION correctly answers the QUESTION, "
    "given the GOLD answer(s) as ground truth.\n\n"
    "Mark it correct when the prediction conveys the same answer as any gold "
    "answer, EVEN IF it is phrased differently, reordered, more verbose, or "
    "surrounded by extra explanation -- as long as the specific thing the "
    "question asks for is present and factually consistent with a gold answer. A "
    "prediction that is a reasonable paraphrase or a differently-scoped but "
    "equivalent form of a gold answer (e.g. 'King James I' for 'King James I of "
    "England') is correct.\n\n"
    "Mark it incorrect when the prediction states a different or wrong fact, "
    "omits an essential qualifier that changes the meaning (e.g. answering "
    "'teak' when the gold answer is '75% of the world's teak', or 'defeated' "
    "when the gold answer is 'regrouped and defeated the Portuguese'), is a "
    "refusal/UNKNOWN, or does not actually answer the question.\n\n"
    "Reply with exactly one word: YES if the prediction is correct, or NO if it "
    "is incorrect. Output nothing else."
)


def _format_gold(gold_answers: List[str]) -> str:
    """Render the gold answer list as a numbered block for the prompt."""
    return "\n".join(f"{i}. {g}" for i, g in enumerate(gold_answers, start=1))


class AnswerJudge:
    """LLM-backed correctness judge returning a boolean per (question, prediction).

    Attributes
    ----------
    generator_module : GeneratorModule
        The generator used as the judge (reuses the configured pipeline model).
    """

    def __init__(self, generator_module):
        self.generator_module = generator_module

    @staticmethod
    def _parse_verdict(text: str) -> bool:
        """Map the model's reply to a boolean. Raises ValueError if unparseable.

        Looks at the first recognizable yes/no token so a stray trailing word
        does not flip the verdict. A reply with neither token is a parse miss and
        is raised to the caller, which falls back to the deterministic metric
        rather than silently guessing.
        """
        lowered = (text or "").strip().lower()
        # Check word-initial tokens first (the instruction asks for a bare YES/NO).
        for token in lowered.replace(".", " ").replace(",", " ").split():
            if token in ("yes", "correct", "true"):
                return True
            if token in ("no", "incorrect", "false"):
                return False
        raise ValueError(f"Unparseable judge verdict: {text!r}")

    def judge(self, question: str, prediction: str, gold_answers: List[str]) -> bool:
        """Return True if the prediction correctly answers the question.

        Raises on an unparseable model reply; callers decide the fallback.
        """
        pred = (prediction or "").strip()
        if not pred:
            return False
        if len(pred) > _MAX_PREDICTION_CHARS:
            pred = pred[:_MAX_PREDICTION_CHARS]
        prompt = (
            f"{_JUDGE_INSTRUCTION}\n\n"
            f"QUESTION: {question}\n\n"
            f"GOLD answer(s):\n{_format_gold(gold_answers)}\n\n"
            f"PREDICTION: {pred}\n\n"
            "Verdict (YES or NO):"
        )
        raw = self.generator_module.generate_response(prompt)
        return self._parse_verdict(raw)


def judge_matches(judge, predictions, references, queries):
    """Per-example judge verdicts as a list of ``Optional[bool]``.

    Each entry is ``True`` (judged correct), ``False`` (judged incorrect), or
    ``None`` when the judge call errored or returned an unparseable reply. A failed
    row is deliberately not substituted with a deterministic exact-match verdict:
    crediting a flaky judge call with the EM result silently inflates the
    ``llm_judge`` metric with a different metric's answer. Callers exclude ``None``
    rows from the accuracy denominator and report the failure count separately, so
    the judge metric reflects only rows the judge actually decided.

    Parameters
    ----------
    judge : AnswerJudge
        The judge to query.
    predictions : List[str]
        Model predictions, one per example.
    references : List[List[str]]
        Gold answers (incl. aliases) per example.
    queries : List[str]
        The question per example -- required for the judge.

    Returns
    -------
    List[Optional[bool]]
    """
    verdicts = []
    for i, (pred, refs, question) in enumerate(zip(predictions, references, queries)):
        try:
            verdicts.append(judge.judge(question, pred, refs))
        except Exception as exc:  # noqa: BLE001 -- never let one row break scoring
            logger.warning("LLM judge failed on example %d (%s); recording as unscored (None)", i, exc)
            verdicts.append(None)
    return verdicts
