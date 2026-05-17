"""
functions.py
============
Talk Move Classifier에서 사용하는 모든 클래스 정의.
논문: Cao et al. (2025) "Enhancing Talk Moves Analysis in Mathematics
      Tutoring through Classroom Teaching Discourse", COLING 2025

Teacher Talk Moves (Tag 0~6):
  0: NONE  1: KPTG  2: GSTUR  3: RESTAT  4: REVOIC  5: PRSACC  6: PRSREA
"""

import json
import time
import pandas as pd
from dataclasses import dataclass
from typing import Optional


# ──────────────────────────────────────────────
# 1. 데이터 구조
# ──────────────────────────────────────────────

@dataclass
class Utterance:
    """대화 한 줄을 표현하는 데이터 클래스."""
    index: int
    turn: int
    speaker: str                           # 'T' (교사)
    sentence: str
    true_tag: Optional[int] = None         # 정답 레이블
    predicted_tag: Optional[int] = None    # 예측 레이블
    confidence: Optional[float] = None     # 예측 신뢰도


# ──────────────────────────────────────────────
# 2. 레이블 정의
# ──────────────────────────────────────────────

class TalkMoveLabels:
    """Teacher Talk Move 레이블 정보."""

    TEACHER = {
        0: "NONE",
        1: "KPTG",
        2: "GSTUR",
        3: "RESTAT",
        4: "REVOIC",
        5: "PRSACC",
        6: "PRSREA",
    }

    @classmethod
    def tag_to_name(cls, tag: Optional[int]) -> str:
        if tag is None:
            return "N/A"
        return cls.TEACHER.get(tag, "?")

    @classmethod
    def name_to_tag(cls, name: str) -> int:
        reverse = {v: k for k, v in cls.TEACHER.items()}
        return reverse.get(name.upper(), 0)


# ──────────────────────────────────────────────
# 3. 프롬프트 생성
# ──────────────────────────────────────────────

class PromptBuilder:
    """LLM 분류용 시스템/유저 프롬프트를 생성하는 클래스."""

    SYSTEM_PROMPT = """You are an expert annotator for mathematics classroom discourse analysis.
You classify teacher utterances using the Talk Move framework (Accountable Talk theory).

## Label Definitions

0 NONE — Does NOT fit any of labels 1-6.
   Apply NONE liberally. Most utterances are NONE.
   NONE includes:
   - Explaining content, giving instructions, managing time/logistics
   - Transitions ("Okay", "Alright", "Let's move on")
   - Affirmations ("Yes", "Good", "Right")
   - Restating what the teacher themselves said (not a student)
   - Describing tasks or learning goals to the whole class
   - Any utterance not directly eliciting student-to-student discourse

1 KPTG (Keeping Everyone Together) — Explicitly directs ALL students to attend to ONE shared idea or answer.
   Key signal: teacher focuses the whole class on a *specific idea just raised*.
   Examples: "Can everyone look at what Maria said?", "Let's all think about that."
   NOT KPTG: general task directions like "Get started", "Please check homework", "Do problem one" → NONE

2 GSTUR (Getting Students to Relate) — Asks a SPECIFIC student to engage with ANOTHER student's idea.
   Key signal: targets one student, references another student's contribution.
   Examples: "John, do you agree with what Sarah said?", "How does that connect to what he said?"

3 RESTAT (Restating) — Teacher repeats or paraphrases a STUDENT's words for the class.
   Key signal: teacher is voicing back what a student said, attributing it to that student.
   Examples: "So what you're saying is...", "She said the answer is 48000."

4 REVOIC (Revoicing) — Teacher re-expresses a student's idea with a slight elaboration or interpretive shift.
   Differs from RESTAT: adds a new framing or interpretation beyond mere repetition.
   Examples: "So you're saying the zeros come from multiplying tens?"

5 PRSACC (Press for Accuracy) — Asks student to CHECK, CORRECT, or VERIFY an answer.
   Key signal: focuses on WHETHER the answer is right/wrong.
   Examples: "Are you sure?", "Is that correct?", "Check that again."
   NOT PRSACC: asking WHY or HOW → that is PRSREA

6 PRSREA (Press for Reasoning) — Asks student to EXPLAIN their thinking, steps, or reasoning.
   Key signal: focuses on HOW or WHY, not just correctness.
   Examples: "Why do you think that?", "How did you get that?", "Talk me through your steps."
   "Talk about what steps you took" → PRSREA (asking for steps/process = reasoning)

## Decision Rules
- When in doubt between a label and NONE → choose NONE.
- KPTG requires directing the whole class to a specific student idea — task instructions alone are NONE.
- PRSACC vs PRSREA: accuracy = "is it right?", reasoning = "why/how?".
- Short phrases ("Start again", "You said what") without clear discourse intent → NONE.

Respond ONLY with JSON: {"tag": <integer 0-6>, "confidence": <float 0.0-1.0>}"""

    @classmethod
    def build_system_prompt(cls) -> str:
        return cls.SYSTEM_PROMPT

    @classmethod
    def build_user_prompt(
        cls,
        utterance: Utterance,
        context: list["Utterance"],
    ) -> str:
        context_block = ""
        if context:
            ctx_text = "\n".join(
                f"  [T]: {u.sentence or '(no text)'}"
                for u in context
            )
            context_block = f"[Prior Context]\n{ctx_text}\n\n"

        return (
            f"{context_block}"
            f"[Target Utterance]\n"
            f"  [T]: {utterance.sentence or '(no text)'}\n\n"
            f"Classify the target utterance. Tag 0 (NONE) if unsure."
        )


# ──────────────────────────────────────────────
# 4. LLM 백엔드 (Anthropic / Gemini)
# ──────────────────────────────────────────────

class _BaseBackend:
    """LLM 백엔드 공통 인터페이스."""
    RETRY_LIMIT = 3
    RETRY_DELAY = 2

    def call(self, system: str, user: str) -> str:
        raise NotImplementedError

    def safe_call(self, system: str, user: str) -> tuple[int, float]:
        for attempt in range(self.RETRY_LIMIT):
            try:
                text = self.call(system, user)
                result = json.loads(text.strip())
                return int(result["tag"]), float(result.get("confidence", 1.0))
            except (json.JSONDecodeError, KeyError, ValueError):
                if attempt < self.RETRY_LIMIT - 1:
                    time.sleep(self.RETRY_DELAY)
        return 0, 0.0


class _AnthropicBackend(_BaseBackend):
    def __init__(self, api_key: Optional[str], model: str):
        from anthropic import Anthropic
        self.client = Anthropic(api_key=api_key) if api_key else Anthropic()
        self.model = model

    def call(self, system: str, user: str) -> str:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=64,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return response.content[0].text


class _GeminiBackend(_BaseBackend):
    def __init__(self, api_key: Optional[str], model: str):
        from google import genai
        self.client = genai.Client(api_key=api_key) if api_key else genai.Client()
        self.model = model

    def call(self, system: str, user: str) -> str:
        from google.genai import types
        response = self.client.models.generate_content(
            model=self.model,
            config=types.GenerateContentConfig(
                system_instruction=system,
                temperature=0.0,
            ),
            contents=user,
        )
        return response.text


class LLMClassifier:
    """
    교사 발화를 분류하는 클래스. Anthropic과 Gemini 백엔드 지원.

    Parameters
    ----------
    provider : "anthropic" | "gemini"
    api_key  : API 키. None이면 환경변수 자동 참조.
    model    : 사용할 모델명. None이면 provider 기본값 사용.
    """

    DEFAULT_MODELS = {
        "anthropic": "claude-haiku-4-5-20251001",
        "gemini":    "gemini-2.0-flash",
    }

    def __init__(
        self,
        provider: str = "gemini",
        api_key: Optional[str] = None,
        model: Optional[str] = None,
    ):
        self.provider = provider.lower()
        if self.provider not in self.DEFAULT_MODELS:
            raise ValueError(
                f"지원하지 않는 provider: {provider!r}. "
                "'anthropic' 또는 'gemini'를 사용하세요."
            )
        self.model = model or self.DEFAULT_MODELS[self.provider]
        self.prompt_builder = PromptBuilder()

        if self.provider == "anthropic":
            self.backend = _AnthropicBackend(api_key, self.model)
        else:
            self.backend = _GeminiBackend(api_key, self.model)

    def classify(
        self,
        utterance: Utterance,
        context: list[Utterance],
    ) -> tuple[int, float]:
        """발화를 분류하고 (predicted_tag, confidence) 반환. 실패 시 (0, 0.0)."""
        system = self.prompt_builder.build_system_prompt()
        user   = self.prompt_builder.build_user_prompt(utterance, context)
        tag, confidence = self.backend.safe_call(system, user)

        if tag not in TalkMoveLabels.TEACHER:
            tag = 0
        return tag, confidence


# ──────────────────────────────────────────────
# 5. 데이터 로더 (교사 발화만)
# ──────────────────────────────────────────────

class DataLoader:
    """xlsx / csv 파일에서 교사 발화만 읽어 Utterance 리스트로 변환."""

    @staticmethod
    def _read_file(path: str, max_rows: Optional[int]) -> pd.DataFrame:
        ext = path.rsplit(".", 1)[-1].lower()
        if ext == "csv":
            return pd.read_csv(path, nrows=max_rows)
        elif ext in ("xlsx", "xls", "xlsm"):
            return pd.read_excel(path, nrows=max_rows)
        else:
            try:
                return pd.read_csv(path, nrows=max_rows)
            except Exception:
                return pd.read_excel(path, nrows=max_rows)

    @staticmethod
    def load(path: str, max_rows: Optional[int] = None) -> list[Utterance]:
        df = DataLoader._read_file(path, max_rows)

        # 교사 발화만 필터링
        if "Speaker" in df.columns:
            df = df[df["Speaker"] == "T"].reset_index(drop=True)

        utterances = []
        for i, row in df.iterrows():
            sentence = row.get("Sentence", "")
            sentence = "" if pd.isna(sentence) else str(sentence)

            tag_val = row.get("Tag")
            true_tag = None if (tag_val is None or pd.isna(tag_val)) else int(tag_val)

            idx_val = row.get("Unnamed: 0", i)
            utterances.append(Utterance(
                index=int(idx_val) if not pd.isna(idx_val) else i,
                turn=int(row["Turn"]) if not pd.isna(row["Turn"]) else 0,
                speaker="T",
                sentence=sentence,
                true_tag=true_tag,
            ))

        return utterances


# ──────────────────────────────────────────────
# 6. 평가 + 혼동행렬
# ──────────────────────────────────────────────

class Evaluator:
    """분류 결과 평가, 리포트 출력, 혼동행렬 시각화."""

    LABELS = ["NONE", "KPTG", "GSTUR", "RESTAT", "REVOIC", "PRSACC", "PRSREA"]

    @staticmethod
    def accuracy(utterances: list[Utterance]) -> float:
        subset = [u for u in utterances if u.predicted_tag is not None and u.true_tag is not None]
        if not subset:
            return 0.0
        correct = sum(1 for u in subset if u.predicted_tag == u.true_tag)
        return correct / len(subset)

    @staticmethod
    def print_report(utterances: list[Utterance]) -> None:
        subset = [u for u in utterances if u.predicted_tag is not None]
        if not subset:
            print("예측 결과가 없습니다.")
            return

        acc = Evaluator.accuracy(utterances)
        print(f"\n{'='*55}")
        print(f"[Teacher Talk Moves]  n={len(subset)},  Accuracy={acc:.1%}")
        print(f"{'='*55}")
        print(f"{'IDX':>5} {'Turn':>5} {'Pred':>8} {'True':>8}  Sentence[:50]")
        print("-"*70)

        for u in subset:
            pred_name = TalkMoveLabels.tag_to_name(u.predicted_tag)
            true_name = TalkMoveLabels.tag_to_name(u.true_tag)
            mark = "✓" if u.predicted_tag == u.true_tag else "✗"
            print(f"{u.index:>5} {u.turn:>5} {pred_name:>8} {true_name:>8}  {mark} {(u.sentence or '')[:50]}")

    @staticmethod
    def plot_confusion_matrix(utterances: list[Utterance]) -> None:
        """혼동행렬을 matplotlib으로 시각화."""
        try:
            import matplotlib.pyplot as plt
            import matplotlib
            import numpy as np
        except ImportError:
            print("matplotlib / numpy가 필요합니다: pip install matplotlib numpy")
            return

        labels = Evaluator.LABELS
        n = len(labels)
        label_to_idx = {l: i for i, l in enumerate(labels)}

        matrix = np.zeros((n, n), dtype=int)
        valid = [u for u in utterances if u.predicted_tag is not None and u.true_tag is not None]

        for u in valid:
            true_name = TalkMoveLabels.tag_to_name(u.true_tag)
            pred_name = TalkMoveLabels.tag_to_name(u.predicted_tag)
            t_idx = label_to_idx.get(true_name, 0)
            p_idx = label_to_idx.get(pred_name, 0)
            matrix[t_idx][p_idx] += 1

        # 사용된 레이블만 표시 (행·열 합계 > 0)
        used = [i for i in range(n) if matrix[i].sum() > 0 or matrix[:, i].sum() > 0]
        matrix = matrix[np.ix_(used, used)]
        tick_labels = [labels[i] for i in used]

        acc = Evaluator.accuracy(utterances)
        fig, ax = plt.subplots(figsize=(8, 6))

        im = ax.imshow(matrix, interpolation="nearest", cmap="Blues")
        plt.colorbar(im, ax=ax)

        ax.set_xticks(range(len(tick_labels)))
        ax.set_yticks(range(len(tick_labels)))
        ax.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=11)
        ax.set_yticklabels(tick_labels, fontsize=11)
        ax.set_xlabel("Predicted", fontsize=12)
        ax.set_ylabel("True", fontsize=12)
        ax.set_title(
            f"Talk Move Confusion Matrix\n"
            f"n={len(valid)},  Accuracy={acc:.1%}",
            fontsize=13, fontweight="bold"
        )

        thresh = matrix.max() / 2.0
        for i in range(len(tick_labels)):
            for j in range(len(tick_labels)):
                val = matrix[i, j]
                if val > 0:
                    ax.text(j, i, str(val),
                            ha="center", va="center", fontsize=12,
                            color="white" if val > thresh else "black")

        plt.tight_layout()
        plt.savefig("confusion_matrix.png", dpi=150, bbox_inches="tight")
        plt.show()
        print("💾 confusion_matrix.png 저장 완료")

    @staticmethod
    def to_dataframe(utterances: list[Utterance]) -> pd.DataFrame:
        records = []
        for u in utterances:
            records.append({
                "index": u.index,
                "turn": u.turn,
                "speaker": u.speaker,
                "sentence": u.sentence,
                "true_tag": u.true_tag,
                "true_label": TalkMoveLabels.tag_to_name(u.true_tag),
                "predicted_tag": u.predicted_tag,
                "predicted_label": TalkMoveLabels.tag_to_name(u.predicted_tag),
                "confidence": u.confidence,
                "correct": u.predicted_tag == u.true_tag if u.true_tag is not None else None,
            })
        return pd.DataFrame(records)


# ──────────────────────────────────────────────
# 7. 파이프라인
# ──────────────────────────────────────────────

class ClassificationPipeline:
    """DataLoader → LLMClassifier → Evaluator 흐름을 조율하는 클래스."""

    def __init__(
        self,
        data_path: str,
        provider: str = "gemini",
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        max_rows: Optional[int] = None,
        context_window: int = 3,
    ):
        self.data_path = data_path
        self.max_rows = max_rows
        self.context_window = context_window
        self.classifier = LLMClassifier(provider=provider, api_key=api_key, model=model)
        self.evaluator = Evaluator()

    def run(self, verbose: bool = True) -> list[Utterance]:
        if verbose:
            print(f"📂 Loading: {self.data_path}")
        utterances = DataLoader.load(self.data_path, self.max_rows)
        if verbose:
            print(f"✅ {len(utterances)} teacher utterances loaded.\n")

        for i, utt in enumerate(utterances):
            context = utterances[max(0, i - self.context_window): i]
            tag, conf = self.classifier.classify(utt, context)
            utt.predicted_tag = tag
            utt.confidence = conf

            if verbose:
                pred_name = TalkMoveLabels.tag_to_name(tag)
                mark = "✓" if tag == utt.true_tag else "✗"
                print(
                    f"[{i+1:>4}/{len(utterances)}] {pred_name:>8}({conf:.2f}) "
                    f"{mark}  {(utt.sentence or '')[:45]}"
                )

        if verbose:
            self.evaluator.print_report(utterances)

        return utterances

    def save_csv(self, utterances: list[Utterance], out_path: Optional[str] = None) -> str:
        if out_path is None:
            base = self.data_path.rsplit(".", 1)[0]
            out_path = base + "_classified.csv"
        df = self.evaluator.to_dataframe(utterances)
        df.to_csv(out_path, index=False)
        print(f"💾 Saved: {out_path}")
        return out_path
