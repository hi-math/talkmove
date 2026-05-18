"""
functions.py
============
Talk Move Classifier — Teacher Only
논문: Cao et al. (2025) COLING

Teacher Talk Moves (Tag 0~6):
  0: NONE  1: KPTG  2: GSTUR  3: RESTAT  4: REVOIC  5: PRSACC  6: PRSREA
"""

import json
import time
import pandas as pd
from dataclasses import dataclass
from typing import Optional
from concurrent.futures import ThreadPoolExecutor, as_completed


# ──────────────────────────────────────────────
# 1. 데이터 구조
# ──────────────────────────────────────────────

@dataclass
class Utterance:
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

    SYSTEM_PROMPT = """You are an expert annotator for mathematics classroom discourse.
Classify the teacher utterance into exactly one of 7 Talk Move labels.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
IMPORTANT: NONE is a last resort. Most utterances belong to labels 1–6.
When uncertain between NONE and another label, choose the other label.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

═══════════════════════════════════════════════
LABEL 1 — KPTG (Keeping Everyone Together)
═══════════════════════════════════════════════
Directing student attention, calling on a specific student by name,
checking shared understanding, managing class participation, or keeping
the whole class engaged together.

KPTG sub-types (all are KPTG):
  A. Calling a student by name alone or with a minimal prompt
     • "Clayton"  • "Chandler"  • "Joel"  • "David"  • "Josh"
     • "Carson you want to bring your book up here"
  B. Confirmation/tag questions directed at the whole class
     • "A twodigit times a onedigit right"
     • "17 isnt a multiple of 10 is it"
     • "2400 is not very close to 240 is it"
     • "It helps to talk yourself through it doesnt it"
     • "So your answer will change a little bit wont it"
     • "Did you forget to add 2 zeros"
     • "Do you revise your thinking"
  C. Soliciting class participation / polling
     • "All right who did pink problems"
     • "Who got all the way to doing it partial product way"
     • "These hands went Matt feeling confident"
     • "Every single person except for You missed that"
     • "Thumbs up when you have the answer"
  D. Short directives / attention calls
     • "Go"  • "Please go check your homework with a partner"
  E. Referencing a student's idea to re-engage the class
     • "Youre comparing"  • "You said what"
     • "It was following a pattern and then it messed you up"
     • "So you did it right underneath the way that they know"

NOT KPTG → NONE: pure logistics not aimed at discourse engagement
  ✗ "You guys should go ask someone else" → NONE (redirecting, not engaging)
  ✗ "I want you to take your whiteboard and swap with your partner" → NONE (task setup)

═══════════════════════════════════════════════
LABEL 2 — GSTUR (Getting Students to Relate)
═══════════════════════════════════════════════
Asking a student to engage with, compare to, or evaluate a classmate's idea.
Key signal: reference to what another student said/did + request to react.

Real examples:
  • "How do you know he is right?"
  • "Any disagreements with how Carson set up her partition rectangle?"
  • "Anyone do it differently?"
  • "Anyone else have a different response?"
  • "What did you do differently?"
  • "Any disagreements on how Matt or Dakota solved the problem?"
  • "Dakota do you have a different answer or did you just do it the same way?"
  • "What did you notice that Faith did while she was revising?"
  • "But as she was working it did she actually do it right?"

NOT GSTUR → KPTG: "Jesse" alone (single student name) = calling on = KPTG

═══════════════════════════════════════════════
LABEL 3 — RESTAT (Restating)
═══════════════════════════════════════════════
Teacher echoes a student's exact words back (even one word/number counts).
Key signal: teacher is repeating what a student just said.

Real examples:
  • "Seven"  • "Two zeros"  • "Multiples"  • "Times 20 thank you"
  • "4 extra zeros to your basic fact"
  • "Were doing a twodigit number times a twodigit number"

NOT RESTAT → NONE:
  ✗ "Its multiplication" (teacher stating, not echoing a student) → NONE

═══════════════════════════════════════════════
LABEL 4 — REVOIC (Revoicing)
═══════════════════════════════════════════════
Teacher paraphrases or elaborates a student's idea with a slight shift.
Key signal: teacher re-expresses student thinking with added framing.

Real examples:
  • "Youre adding"  • "Also 3"  • "You counted by 5"
  • "300 sir youre correct"  • "Just 28 because there are no extra zeros"
  • "When you started working it the second time you realized..."

NOT REVOIC → NONE (teacher narrating their own observation, not a student's idea):
  ✗ "I heard you say 56 and thinking through" → NONE (teacher thinking aloud)
  ✗ "I heard that 24000" → NONE (teacher noting what they overheard, no elaboration)

═══════════════════════════════════════════════
LABEL 5 — PRSACC (Pressing for Accuracy)
═══════════════════════════════════════════════
Eliciting a math answer, fact, term, calculation, or step from a student.
This is the broadest active label. Applies whenever teacher asks for any
mathematical content — even a single expression or formula.

Real examples (NOTE: bare math expressions count as PRSACC):
  • "4 times 50"  • "30 times 7"  • "70 times 8"  • "20 times 40"
  • "Basic facts with what"  • "Talk about what steps you took to get there"
  • "Anyone feel pretty confident about how they did it"
  • "I hear 81000 Whos correct"
  • "How many extra zeros are in the problem"
  • "Jesse whats different about this problem"
  • "Explain please"

NOT PRSACC → NONE:
  ✗ "Basic fact in there" (teacher statement, not a question) → NONE
  ✗ "I have no idea what you just said" (teacher reaction) → NONE

═══════════════════════════════════════════════
LABEL 6 — PRSREA (Pressing for Reasoning)
═══════════════════════════════════════════════
Asking a student to explain WHY or justify their thinking.
Key signal: contains "why" or equivalent reasoning request.
Even single-word "Why" counts as PRSREA.

Real examples:
  • "Why"  • "Why Clayton"  • "Why not 80"
  • "Why 48000 and not 4800"  • "Why Matt why not"
  • "Why is it not 81000"
  • "Why are you just automatically changing your answer"

═══════════════════════════════════════════════
LABEL 0 — NONE (None of the above)
═══════════════════════════════════════════════
Pure content delivery, logistics, transitions, or filler with no
interactive discourse function aimed at engaging students.

Real examples of genuine NONE:
  • "Get started on mental math"  • "You have about another minute and a half"
  • "Making sure that you can see how your answers should be the same"
  • "Youre mental math youre just doing quick extended facts"
  • "I want you to take your whiteboard and swap with your partner" (task setup)
  • "You guys should go ask someone else" (redirect, not engagement)
  • "I heard you say 56 and thinking through" (teacher thinking aloud)
  • "I heard that 24000" (narrating observation without elaborating)
  • "Basic fact in there" (statement, not a question or prompt)
  • "Its multiplication" (teacher labeling, not echoing a student)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DECISION GUIDE (in order)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. Contains "why" or justification request?                    → PRSREA
2. Asks for math content, answer, formula, or steps?           → PRSACC
   (bare math expression like "4 times 50" = PRSACC)
3. References a classmate's idea and asks student to react?   → GSTUR
4. Teacher echoing student's exact words?                     → RESTAT
5. Teacher reframing/elaborating student's idea?              → REVOIC
6. Student name alone / tag question / participation poll
   / attention direction / short directive?                    → KPTG
7. Pure logistics / content delivery / teacher statement?      → NONE

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
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
            "Use the decision guide. Prefer labels 1-6 over NONE when uncertain.\n"
            "Output JSON only."
        )


# ──────────────────────────────────────────────
# 4. LLM 백엔드
# ──────────────────────────────────────────────

class _BaseBackend:
    RETRY_LIMIT = 5
    RETRY_DELAY = 3

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
            except Exception:
                if attempt < self.RETRY_LIMIT - 1:
                    time.sleep(self.RETRY_DELAY * (attempt + 1))
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
    """교사 발화 분류기. Anthropic / Gemini 백엔드 지원."""

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
# 5. 데이터 로더
# ──────────────────────────────────────────────

class DataLoader:
    """xlsx / csv 파일에서 교사 발화만 읽어 Utterance 리스트로 변환."""

    @staticmethod
    def _read_file(path: str) -> pd.DataFrame:
        """확장자에 따라 전체 파일을 읽기 (샘플링은 load()에서 처리)."""
        ext = path.rsplit(".", 1)[-1].lower()
        if ext == "csv":
            return pd.read_csv(path)
        elif ext in ("xlsx", "xls", "xlsm"):
            return pd.read_excel(path)
        else:
            try:
                return pd.read_csv(path)
            except Exception:
                return pd.read_excel(path)

    @staticmethod
    def load(
        path: str,
        sample_n: Optional[int] = None,
        random_state: int = 42,
    ) -> list[Utterance]:
        """
        교사 발화만 로드. sample_n이 주어지면 전체에서 랜덤 샘플링.
        (앞에서 자르는 max_rows 방식이 아니므로 데이터 리크 없음)
        """
        df = DataLoader._read_file(path)

        if "Speaker" in df.columns:
            df = df[df["Speaker"] == "T"].reset_index(drop=True)

        # 랜덤 샘플링 (sample_n이 전체보다 크면 전체 사용)
        if sample_n is not None and sample_n < len(df):
            df = df.sample(n=sample_n, random_state=random_state).reset_index(drop=True)
            print(f"🎲 랜덤 샘플링: {sample_n}개 / 전체 교사 발화에서 추출 (seed={random_state})")

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

    LABELS = ["NONE", "KPTG", "GSTUR", "RESTAT", "REVOIC", "PRSACC", "PRSREA"]

    @staticmethod
    def accuracy(utterances: list[Utterance]) -> float:
        subset = [u for u in utterances if u.predicted_tag is not None and u.true_tag is not None]
        if not subset:
            return 0.0
        return sum(1 for u in subset if u.predicted_tag == u.true_tag) / len(subset)

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
            mark = "✓" if u.predicted_tag == u.true_tag else ("✗" if u.true_tag is not None else " ")
            print(f"{u.index:>5} {u.turn:>5} {pred_name:>8} {true_name:>8}  {mark} {(u.sentence or '')[:50]}")

    @staticmethod
    def print_distribution(utterances: list[Utterance]) -> None:
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
    def print_per_label_metrics(utterances: list[Utterance]) -> None:
        import numpy as np
        labels = Evaluator.LABELS
        n = len(labels)
        label_to_idx = {l: i for i, l in enumerate(labels)}
        matrix = np.zeros((n, n), dtype=int)

        valid = [u for u in utterances if u.predicted_tag is not None and u.true_tag is not None]
        for u in valid:
            t = label_to_idx.get(TalkMoveLabels.tag_to_name(u.true_tag), 0)
            p = label_to_idx.get(TalkMoveLabels.tag_to_name(u.predicted_tag), 0)
            matrix[t][p] += 1

        print(f"\n{'Label':<8} {'Total':>6} {'Recall':>8} {'Precision':>10} {'F1':>7}")
        print("-" * 45)
        for i, label in enumerate(labels):
            total = matrix[i].sum()
            correct = matrix[i][i]
            pred_total = matrix[:, i].sum()
            recall = correct / total if total > 0 else 0
            prec   = correct / pred_total if pred_total > 0 else 0
            f1     = 2 * prec * recall / (prec + recall) if (prec + recall) > 0 else 0
            print(f"{label:<8} {total:>6} {recall:>7.1%} {prec:>9.1%} {f1:>7.3f}")

    @staticmethod
    def plot_confusion_matrix(utterances: list[Utterance]) -> None:
        try:
            import matplotlib.pyplot as plt
            import numpy as np
        except ImportError:
            print("pip install matplotlib numpy")
            return

        labels = Evaluator.LABELS
        n = len(labels)
        label_to_idx = {l: i for i, l in enumerate(labels)}
        matrix = np.zeros((n, n), dtype=int)

        valid = [u for u in utterances if u.predicted_tag is not None and u.true_tag is not None]
        for u in valid:
            t = label_to_idx.get(TalkMoveLabels.tag_to_name(u.true_tag), 0)
            p = label_to_idx.get(TalkMoveLabels.tag_to_name(u.predicted_tag), 0)
            matrix[t][p] += 1

        used = [i for i in range(n) if matrix[i].sum() > 0 or matrix[:, i].sum() > 0]
        m = matrix[np.ix_(used, used)]
        tick_labels = [labels[i] for i in used]

        acc = Evaluator.accuracy(utterances)
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(m, interpolation="nearest", cmap="Blues")
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
        thresh = m.max() / 2.0
        for i in range(len(tick_labels)):
            for j in range(len(tick_labels)):
                val = m[i, j]
                if val > 0:
                    ax.text(j, i, str(val), ha="center", va="center",
                            fontsize=12, color="white" if val > thresh else "black")
        plt.tight_layout()
        plt.savefig("confusion_matrix.png", dpi=150, bbox_inches="tight")
        plt.show()
        print("💾 confusion_matrix.png 저장 완료")

    @staticmethod
    def to_dataframe(utterances: list[Utterance]) -> pd.DataFrame:
        return pd.DataFrame([{
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
        } for u in utterances])


# ──────────────────────────────────────────────
# 7. 파이프라인
# ──────────────────────────────────────────────

class ClassificationPipeline:

    def __init__(
        self,
        data_path: str,
        provider: str = "gemini",
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        sample_n: Optional[int] = None,
        random_state: int = 42,
        context_window: int = 3,
        workers: int = 5,
    ):
        self.data_path = data_path
        self.sample_n = sample_n
        self.random_state = random_state
        self.context_window = context_window
        self.workers = workers
        self.classifier = LLMClassifier(provider=provider, api_key=api_key, model=model)
        self.evaluator = Evaluator()

    def _classify_one(self, args):
        i, utt, context = args
        tag, conf = self.classifier.classify(utt, context)
        return i, tag, conf

    def run(self, verbose: bool = True) -> list[Utterance]:
        if verbose:
            print(f"📂 Loading: {self.data_path}")
        utterances = DataLoader.load(self.data_path, self.sample_n, self.random_state)
        n = len(utterances)
        if verbose:
            print(f"✅ {n} teacher utterances loaded.")
            print(f"🚀 병렬 처리 시작 (workers={self.workers})\n")

        tasks = [
            (i, utt, utterances[max(0, i - self.context_window): i])
            for i, utt in enumerate(utterances)
        ]

        completed = 0
        with ThreadPoolExecutor(max_workers=self.workers) as executor:
            futures = {executor.submit(self._classify_one, t): t[0] for t in tasks}
            for future in as_completed(futures):
                i, tag, conf = future.result()
                utterances[i].predicted_tag = tag
                utterances[i].confidence = conf
                completed += 1
                if verbose:
                    utt = utterances[i]
                    pred_name = TalkMoveLabels.tag_to_name(tag)
                    mark = "✓" if tag == utt.true_tag else ("✗" if utt.true_tag is not None else " ")
                    print(f"[{completed:>4}/{n}] {pred_name:>8}({conf:.2f}) {mark}  {(utt.sentence or '')[:45]}")

        if verbose:
            self.evaluator.print_distribution(utterances)
            self.evaluator.print_per_label_metrics(utterances)
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
