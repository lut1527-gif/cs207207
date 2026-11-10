"""Bayes' Theorem:  P(A|B) = P(B|A) * P(A) / P(B)"""
from typing import Dict, List


def _check_probability(name: str, p: float) -> None:
    if p is None or not (0.0 <= p <= 1.0):
        raise ValueError(f"{name} ต้องอยู่ระหว่าง 0 ถึง 1 (ได้รับ {p})")


def calculate_evidence(prior: float, likelihood: float, likelihood_not: float) -> float:
    """Law of Total Probability: P(B) = P(B|A)P(A) + P(B|¬A)P(¬A)"""
    _check_probability("P(A)", prior)
    _check_probability("P(B|A)", likelihood)
    _check_probability("P(B|¬A)", likelihood_not)
    return likelihood * prior + likelihood_not * (1 - prior)


def calculate_posterior(prior: float, likelihood: float, evidence: float) -> float:
    """คืนค่า Posterior P(A|B)"""
    _check_probability("Prior P(A)", prior)
    _check_probability("Likelihood P(B|A)", likelihood)
    _check_probability("Evidence P(B)", evidence)
    if evidence == 0:
        raise ValueError("Evidence P(B) = 0 ทำให้หารด้วยศูนย์ ไม่สามารถคำนวณได้")

    posterior = likelihood * prior / evidence
    if posterior > 1 + 1e-9:
        raise ValueError(
            "ค่าที่กรอกขัดแย้งกัน: P(B) ต้องไม่น้อยกว่า P(B|A)·P(A) "
            f"(= {likelihood * prior:.4f}) เพราะจะได้ Posterior > 1"
        )
    return min(posterior, 1.0)


def interpret(prior: float, posterior: float) -> str:
    """สรุปผลเป็นข้อความ"""
    change = posterior - prior
    if abs(change) < 1e-9:
        return "หลักฐาน B ไม่ได้เปลี่ยนความเชื่อเกี่ยวกับ A เลย (A และ B เป็นอิสระต่อกัน)"
    ratio = posterior / prior if prior > 0 else float("inf")
    direction = "เพิ่มขึ้น" if change > 0 else "ลดลง"
    return (
        f"เมื่อเจอหลักฐาน B ความน่าจะเป็นของ A {direction} "
        f"จาก {prior:.2%} เป็น {posterior:.2%} (เปลี่ยนไป {ratio:.2f} เท่า)"
    )


def bayes_report(prior: float, likelihood: float, evidence: float) -> Dict:
    posterior = calculate_posterior(prior, likelihood, evidence)
    return {
        "prior": prior,
        "likelihood": likelihood,
        "evidence": evidence,
        "posterior": posterior,
        "posterior_not": 1 - posterior,
        "numerator": likelihood * prior,
        "lift": (posterior / prior) if prior > 0 else None,
        "summary": interpret(prior, posterior),
    }


def sensitivity_curve(likelihood: float, likelihood_not: float, points: int = 101) -> Dict[str, List[float]]:
    """Posterior เมื่อเปลี่ยน Prior ตั้งแต่ 0-1 (ใช้ P(B) จาก Total Probability)"""
    priors, posteriors = [], []
    for i in range(points):
        p = i / (points - 1)
        evidence = calculate_evidence(p, likelihood, likelihood_not)
        priors.append(round(p, 4))
        posteriors.append(likelihood * p / evidence if evidence > 0 else 0.0)
    return {"prior": priors, "posterior": posteriors}


def natural_frequencies(prior: float, likelihood: float, likelihood_not: float, population: int = 10000) -> List[Dict]:
    """อธิบายด้วยจำนวนคน: ใน N คน มีกี่คนที่ A จริง/ไม่จริง และตรวจพบ B"""
    a_true = population * prior
    a_false = population - a_true
    b_and_a = a_true * likelihood
    b_and_not_a = a_false * likelihood_not
    return [
        {"กลุ่ม": "A จริง", "จำนวน": round(a_true, 1), "พบ B": round(b_and_a, 1), "ไม่พบ B": round(a_true - b_and_a, 1)},
        {"กลุ่ม": "A ไม่จริง", "จำนวน": round(a_false, 1), "พบ B": round(b_and_not_a, 1), "ไม่พบ B": round(a_false - b_and_not_a, 1)},
    ]
