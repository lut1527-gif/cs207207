"""สถิติเชิงพรรณนา: Mean, Median, Mode, SD, Variance (+ Quartile, Range, Histogram)"""
import math
import re
from typing import Dict, List

import numpy as np

from services.cal_mode import calculate_mode, frequency_table


def parse_numbers(text: str) -> List[float]:
    """แปลงข้อความ เช่น '-100, 0, 10 20; 30' ให้เป็น list ของตัวเลข"""
    if text is None or not text.strip():
        raise ValueError("กรุณากรอกตัวเลขอย่างน้อย 1 ค่า")

    tokens = [t for t in re.split(r"[,\s;]+", text.strip()) if t]
    values: List[float] = []
    for token in tokens:
        try:
            number = float(token)
        except ValueError:
            raise ValueError(f"'{token}' ไม่ใช่ตัวเลข") from None
        if math.isnan(number) or math.isinf(number):
            raise ValueError(f"ไม่รองรับค่า '{token}'")
        values.append(number)
    return values


def calculate_mean(values: List[float]) -> float:
    return sum(values) / len(values)


def calculate_median(values: List[float]) -> float:
    ordered = sorted(values)
    n = len(ordered)
    mid = n // 2
    if n % 2 == 1:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2


def calculate_variance(values: List[float], sample: bool = False) -> float:
    """Variance: ประชากร (หาร N) หรือ ตัวอย่าง (หาร N-1)"""
    n = len(values)
    if sample and n < 2:
        raise ValueError("Sample variance/SD ต้องมีข้อมูลอย่างน้อย 2 ค่า")
    mean = calculate_mean(values)
    squared = sum((x - mean) ** 2 for x in values)
    return squared / (n - 1 if sample else n)


def calculate_sd(values: List[float], sample: bool = False) -> float:
    return math.sqrt(calculate_variance(values, sample))


def histogram_data(values: List[float], bins: int = 6) -> Dict[str, List[float]]:
    """ข้อมูลฮิสโตแกรม: จุดกึ่งกลางของแต่ละช่วง + ความถี่"""
    counts, edges = np.histogram(values, bins=bins)
    midpoints = [(edges[i] + edges[i + 1]) / 2 for i in range(len(counts))]
    return {
        "midpoint": [round(float(m), 4) for m in midpoints],
        "count": [int(c) for c in counts],
    }


def summarize(values: List[float], sample: bool = False) -> Dict:
    """สรุปค่าสถิติทั้งหมดในครั้งเดียว (UI เรียกฟังก์ชันนี้ฟังก์ชันเดียว)"""
    if not values:
        raise ValueError("ไม่มีข้อมูลสำหรับคำนวณ")

    ordered = sorted(values)
    q1, q3 = np.percentile(ordered, [25, 75])
    mode_info = calculate_mode(values)

    return {
        "count": len(values),
        "mean": calculate_mean(values),
        "median": calculate_median(values),
        "mode": mode_info,
        "variance": calculate_variance(values, sample),
        "sd": calculate_sd(values, sample),
        "min": ordered[0],
        "max": ordered[-1],
        "range": ordered[-1] - ordered[0],
        "q1": float(q1),
        "q3": float(q3),
        "iqr": float(q3 - q1),
        "sum": sum(values),
        "sorted": ordered,
        "frequency": frequency_table(values),
        "is_sample": sample,
    }
