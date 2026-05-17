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
    speaker: str
    sentence: str
    true_tag: Optional[int] = None
    predicted_tag: Optional[int] = None
    confidence: Optional[float] = None


# ──────────────────────────────────────────────
# 2. 레이블 정의
# ──────────────────────────────────────────────

class TalkMoveLabels:
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

    SYSTEM_PROMPT = """You are an expert annotator for mathematics classroom discourse.
Classify the teacher utterance into exactly one of the 7 Talk Move labels below.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
LABEL DEFINITIONS, DESCRIPTIONS, AND EXAMPLES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1  KPTG  Keeping Everyone Together
   Prompting students to be active listeners and orienting students to each other.
   The teacher directs the whole class to pay attention to a specific student's idea.
   ✓ "What did Eliza just say her equation was?"
   ✓ "Can everyone look at what Marcus put on the board?"
   ✓ "Who can tell me what John just said?"
   ✓ "Did everyone hear that? Say it again for the class."
   — Also applies when teacher asks the class to check or share with a partner:
   ✓ "Please go check your homework with a partner."

2  GSTUR  Getting Students to Relate to Another's Ideas
   Prompting students to react to what a classmate said.
   The teacher explicitly asks one student to engage with another student's idea.
   ✓ "Do you agree with Juan that the answer is 7/10?"
   ✓ "How does that connect to what Maria said?"
   ✓ "How do you know he is right?"

3  RESTAT  Restating
   Repeating all or part of what a student said word for word.
   The teacher echoes a student's exact words back to the class.
   ✓ "Add two here." (repeating student's words)
   ✓ "So she said the answer is 48."
   ✓ "You're saying X is the number of cars."

4  REVOIC  Revoicing
   Repeating what a student said but adding on or changing the wording.
   The teacher paraphrases or reframes a student's idea with elaboration.
   ✓ "Julia told us she would add two here." (adds attribution + slight elaboration)
   ✓ "So you're saying the zeros come from multiplying by tens?"
   ✓ "You're comparing the two methods." (reframes student's action)

5  PRSACC  Pressing for Accuracy
   Prompting students to make a mathematical contribution or use mathematical language.
   Focuses on correctness, precision, or proper math vocabulary.
   ✓ "Can you give an example of an ordered pair?"
   ✓ "Is that the right term?"
   ✓ "Can you say that using math vocabulary?"
   ✓ "Are you sure that's correct?"
   ✓ "Start again." (prompting student to redo for correctness)
   ✓ "You said what?" (signaling inaccuracy, prompting correction)

6  PRSREA  Pressing for Reasoning
   Prompting students to explain, provide evidence, share their thinking behind
   a decision, or connect ideas or representations.
   Focuses on WHY or HOW, not just whether the answer is right.
   ✓ "Why could I argue that the slope should be increasing?"
   ✓ "Why are you just automatically changing your answer?"
   ✓ "Talk about what steps you took to get there."
   ✓ "How did you get that?"
   ✓ "Can you explain your thinking?"

0  NONE  None of the above
   Use NONE only when the utterance clearly does not fit labels 1–6.
   NONE includes:
   - Content explanation or instruction NOT tied to a specific student's idea
   - Pure logistics / time management ("You have one more minute")
   - Filler / acknowledgment only ("Okay", "Alright", "I'm sorry")
   - Task setup describing what students will do ("There's a partitioned rectangle problem...")
   - Learning objective statements ("Your learning intention is you can multiply...")
   ✗ Do NOT use NONE for short or vague utterances if they clearly signal
     a press for accuracy or reasoning.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DECISION GUIDE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Step 1. Does the teacher orient students to LISTEN TO / REACT TO a classmate?
        → Yes, listen/attend     → KPTG
        → Yes, react/engage      → GSTUR

Step 2. Does the teacher repeat a student's words?
        → Exact repeat           → RESTAT
        → Paraphrase/elaborate   → REVOIC

Step 3. Does the teacher press a student?
        → For correctness/vocab  → PRSACC
        → For explanation/why    → PRSREA

Step 4. None of the above       → NONE

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
IMPORTANT NOTES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- Short utterances CAN be PRSACC or PRSREA if the intent is clear from context.
- "Start again", "You said what?" → PRSACC (pressing for corrected/accurate response)
- "Talk about your steps", "Why did you do that?" → PRSREA
- Describing task instructions to the whole class (not referencing a student's idea) → NONE
- KPTG does NOT require a question; a directive like "Go check with a partner" qualifies.

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
            f"Classify using the decision guide. Output JSON only."
        )


# ──────────────────────────────────────────────
# 4. LLM 백엔드 (Anthropic / Gemini)
# ──────────────────────────────────────────────

class _BaseBackend:
    RETRY_LIMIT = 5
    RETRY_DELAY = 5

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
    def print_distribution(utterances: list[Utterance]) -> None:
        """True vs Predicted 레이블 분포 출력."""
        from collections import Counter
        true_counts = Counter(u.true_tag for u in utterances if u.true_tag is not None)
        pred_counts = Counter(u.predicted_tag for u in utterances if u.predicted_tag is not None)

        print(f"\n{'Tag':<5} {'Label':<8} {'True':>6} {'Pred':>6}  {'Diff':>6}")
        print("-" * 38)
        for tag in range(7):
            name = TalkMoveLabels.tag_to_name(tag)
            t = true_counts.get(tag, 0)
            p = pred_counts.get(tag, 0)
            diff = p - t
            sign = "+" if diff > 0 else ""
            print(f"  {tag}    {name:<8} {t:>6} {p:>6}  {sign}{diff:>5}")

    @staticmethod
    def plot_confusion_matrix(utterances: list[Utterance]) -> None:
        """혼동행렬을 matplotlib으로 시각화."""
        try:
            import matplotlib.pyplot as plt
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
            f"Talk Move Confusion Matrix\nn={len(valid)},  Accuracy={acc:.1%}",
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
                mark = "✓" if tag == utt.true_tag else ("✗" if utt.true_tag is not None else " ")
                print(
                    f"[{i+1:>4}/{len(utterances)}] {pred_name:>8}({conf:.2f}) "
                    f"{mark}  {(utt.sentence or '')[:45]}"
                )

        if verbose:
            self.evaluator.print_distribution(utterances)
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
