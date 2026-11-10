"""โหลดไฟล์จากโฟลเดอร์ data/ (ให้ UI เรียกใช้แทนการอ่านไฟล์เอง)"""
import json
from pathlib import Path
from typing import Dict, List

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
IMG_DIR = DATA_DIR / "imgs"


def load_json(filename: str) -> Dict:
    path = DATA_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"ไม่พบไฟล์ {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_raw_data() -> Dict:
    return load_json("raw_data.json")


def get_text_datasets() -> List[Dict]:
    return load_raw_data().get("text_datasets", [])


def get_number_sets() -> List[Dict]:
    return load_raw_data().get("number_sets", [])


def text_datasets_table() -> List[Dict]:
    """สรุปชุดข้อความแต่ละชุดเป็นตาราง"""
    rows = []
    for d in get_text_datasets():
        words = d["text"].split()
        rows.append({
            "id": d["id"],
            "title": d["title"],
            "language": d.get("language", "-"),
            "words": len(words),
            "unique_words": len(set(w.strip('.,').lower() for w in words)),
            "preview": d["text"][:80] + "...",
        })
    return rows


def number_sets_table() -> List[Dict]:
    rows = []
    for s in get_number_sets():
        v = s["values"]
        rows.append({"id": s["id"], "title": s["title"], "n": len(v), "min": min(v), "max": max(v), "values": ", ".join(str(x) for x in v)})
    return rows


def list_data_files() -> List[Dict]:
    """รายชื่อไฟล์ทั้งหมดใน data/ (รวมโฟลเดอร์ย่อย)"""
    rows = []
    for p in sorted(DATA_DIR.rglob("*")):
        if p.is_file():
            rows.append({
                "file": str(p.relative_to(DATA_DIR)),
                "type": p.suffix.lstrip(".") or "-",
                "size_kb": round(p.stat().st_size / 1024, 2),
            })
    return rows


def load_members() -> Dict:
    """ข้อมูลสมาชิก + path รูปภาพ (image_path เป็น None ถ้าไม่พบไฟล์)"""
    data = load_json("members.json")
    for m in data["members"]:
        img = IMG_DIR / m.get("image", "")
        m["image_path"] = str(img) if img.is_file() else None
    return data
