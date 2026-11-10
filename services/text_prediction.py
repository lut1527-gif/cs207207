"""Markov Chain สำหรับทำนายคำถัดไป (รองรับ order 1-3)"""
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import pandas as pd

_TOKEN = re.compile(r"[\w\u0E00-\u0E7F']+")


def tokenize(text: str) -> List[str]:
    return _TOKEN.findall(text.lower())


@dataclass
class MarkovModel:
    order: int
    counts: Dict[Tuple[str, ...], Counter] = field(default_factory=dict)
    unigrams: Counter = field(default_factory=Counter)
    tokens: List[str] = field(default_factory=list)

    @property
    def vocabulary_size(self) -> int:
        return len(self.unigrams)

    @property
    def n_states(self) -> int:
        return len(self.counts)


def build_model(text: str, order: int = 1) -> MarkovModel:
    """สร้างตารางนับ transition จากข้อความ: state (n คำ) -> คำถัดไป"""
    if order < 1:
        raise ValueError("order ต้องมีค่าอย่างน้อย 1")
    tokens = tokenize(text)
    if len(tokens) <= order:
        raise ValueError(f"ข้อความสั้นเกินไป (มี {len(tokens)} คำ ต้องมากกว่า {order} คำ)")

    counts: Dict[Tuple[str, ...], Counter] = defaultdict(Counter)
    for i in range(len(tokens) - order):
        state = tuple(tokens[i:i + order])
        counts[state][tokens[i + order]] += 1

    return MarkovModel(order=order, counts=dict(counts), unigrams=Counter(tokens), tokens=tokens)


def transition_probabilities(model: MarkovModel, state: Tuple[str, ...]) -> Dict[str, float]:
    """P(next | state) = count(state, next) / count(state)"""
    nexts = model.counts.get(state)
    if not nexts:
        return {}
    total = sum(nexts.values())
    return {word: c / total for word, c in nexts.items()}


def _resolve_state(model: MarkovModel, context: str) -> Tuple[Tuple[str, ...], bool]:
    """หา state จากข้อความอ้างอิง; ถ้าไม่เจอให้ถอยไปใช้คำสุดท้าย (backoff)"""
    words = tokenize(context)
    if not words:
        raise ValueError("กรุณากรอกข้อความอ้างอิงเริ่มต้นอย่างน้อย 1 คำ")
    state = tuple(words[-model.order:])
    if state in model.counts:
        return state, False
    # backoff: หา state ที่ลงท้ายด้วยคำสุดท้ายของข้อความ
    last = words[-1]
    candidates = [s for s in model.counts if s[-1] == last]
    if candidates:
        return max(candidates, key=lambda s: sum(model.counts[s].values())), True
    return (), True


def predict_next(model: MarkovModel, context: str, top_k: int = 5) -> Dict:
    """ทำนายคำถัดไป top_k อันดับ พร้อมความน่าจะเป็น"""
    state, backoff = _resolve_state(model, context)
    probs = transition_probabilities(model, state)
    if not probs:
        raise ValueError("ไม่พบคำนี้ในข้อมูลอ้างอิง ลองใช้คำอื่นที่อยู่ในข้อความ")
    ranked = sorted(probs.items(), key=lambda kv: (-kv[1], kv[0]))[:top_k]
    return {
        "state": " ".join(state),
        "used_backoff": backoff,
        "predictions": [
            {"word": w, "probability": p, "count": model.counts[state][w]}
            for w, p in ranked
        ],
    }


def generate_text(model: MarkovModel, seed_text: str, length: int = 15, random_seed: int = 42) -> str:
    """สุ่มเดินตาม Markov Chain เพื่อสร้างประโยคต่อจากข้อความเริ่มต้น"""
    rng = random.Random(random_seed)
    words = tokenize(seed_text)
    if not words:
        raise ValueError("กรุณากรอกข้อความเริ่มต้นอย่างน้อย 1 คำ")
    output = list(words)

    for _ in range(length):
        state, _ = _resolve_state(model, " ".join(output))
        probs = transition_probabilities(model, state)
        if not probs:
            break
        choices, weights = zip(*sorted(probs.items()))
        output.append(rng.choices(choices, weights=weights, k=1)[0])
    return " ".join(output)


def transition_matrix(model: MarkovModel, max_states: int = 10, max_words: int = 10) -> pd.DataFrame:
    """Transition Probability Matrix (แถว = state ปัจจุบัน, คอลัมน์ = คำถัดไป)
    แสดงเฉพาะ state / คำที่พบบ่อยที่สุด เพื่อให้อ่านง่าย"""
    state_totals = {s: sum(c.values()) for s, c in model.counts.items()}
    top_states = [s for s, _ in sorted(state_totals.items(), key=lambda kv: (-kv[1], kv[0]))[:max_states]]

    next_totals: Counter = Counter()
    for s in top_states:
        next_totals.update(model.counts[s])
    top_words = [w for w, _ in sorted(next_totals.items(), key=lambda kv: (-kv[1], kv[0]))[:max_words]]

    data = []
    for s in top_states:
        total = state_totals[s]
        data.append([round(model.counts[s].get(w, 0) / total, 3) for w in top_words])
    return pd.DataFrame(data, index=[" ".join(s) for s in top_states], columns=top_words)


def model_summary(model: MarkovModel) -> Dict:
    return {
        "order": model.order,
        "total_words": len(model.tokens),
        "vocabulary_size": model.vocabulary_size,
        "n_states": model.n_states,
        "top_words": model.unigrams.most_common(5),
    }
