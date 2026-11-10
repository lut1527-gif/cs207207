"""คำนวณฐานนิยม (Mode) — รองรับกรณีมีหลายฐานนิยม (multimodal)"""
from collections import Counter
from typing import Dict, List


def frequency_table(values: List[float]) -> List[Dict]:
    """ตารางความถี่ของค่าแต่ละค่า เรียงจากมากไปน้อย"""
    counts = Counter(values)
    total = len(values)
    rows = [
        {"value": v, "count": c, "percent": round(c / total * 100, 2)}
        for v, c in counts.items()
    ]
    rows.sort(key=lambda r: (-r["count"], r["value"]))
    return rows


def calculate_mode(values: List[float]) -> Dict:
    """
    คืนค่า {"modes": [...], "frequency": int, "has_mode": bool}
    ถ้าทุกค่าไม่ซ้ำกันเลย (และมีมากกว่า 1 ค่า) ถือว่า "ไม่มีฐานนิยม"
    """
    if not values:
        raise ValueError("ไม่มีข้อมูลสำหรับคำนวณฐานนิยม")

    counts = Counter(values)
    max_count = max(counts.values())

    if max_count == 1 and len(counts) > 1:
        return {"modes": [], "frequency": 1, "has_mode": False}

    modes = sorted(v for v, c in counts.items() if c == max_count)
    return {"modes": modes, "frequency": max_count, "has_mode": True}
