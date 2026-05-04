"""Debug latest Q4 matching test (student 0012346900, exam morj6pbmpfepg)."""
import sys
sys.path.insert(0, "D:/repos/exam-backend")
import cv2
import json
import preprocessing
import handwriting
from handwriting import _classify_letter

ALIGNED = "D:/repos/exam-backend/output/0012346900_page1.jpg"
MAP = "C:/Users/faruk/AppData/Roaming/iku-exam-generator/exams/morj6pbmpfepg.map.json"

img = cv2.imread(ALIGNED)
m = json.load(open(MAP, "r", encoding="utf-8"))

# Find Q4 in any page
q4 = None
for page in m.get("pages", []):
    qs = page.get("questions", {})
    if "4" in qs:
        q4 = qs["4"]
        break

if q4 is None:
    print("No Q4 in map")
    sys.exit(1)

print(f"Q4 type: {q4.get('type')}")
correct = q4.get("expectedAnswer", {}).get("correctMatches", {})
print(f"correctMatches: {correct}\n")

allowed = {str(v).upper() for v in correct.values() if isinstance(v, str) and len(v) == 1 and v.isalpha()}
print(f"allowed_set: {allowed}\n")

answer_boxes = q4.get("answerBoxes", {})
for idx in sorted(answer_boxes.keys()):
    box = answer_boxes[idx]
    expected_letter = correct.get(idx, "?")
    print(f"=== Box {idx} (expected={expected_letter}) ===")
    print(f"  box: {box}")

    x, y, w, h = int(box["x"]), int(box["y"]), int(box["w"]), int(box["h"])
    raw = img[y:y + h, x:x + w]
    cv2.imwrite(f"D:/repos/exam-backend/output/_latest_q4_box{idx}_raw.png", raw)

    for inset in (0, 1, 2, 3, 4, 5, 6, 7, 8):
        try:
            crop = preprocessing.crop_for_letter_cnn(img, box, inset=inset)
        except Exception as e:
            print(f"  inset={inset}: ERR {e}")
            continue
        cv2.imwrite(
            f"D:/repos/exam-backend/output/_latest_q4_box{idx}_cnn_inset{inset}.png",
            cv2.resize(crop, (140, 140), interpolation=cv2.INTER_NEAREST) if crop.size else crop,
        )
        sum_p = int(crop.sum()) if crop.size else 0
        r_allowed = _classify_letter(crop, allowed=allowed) if sum_p > 100 else None
        r_full = _classify_letter(crop, allowed=None) if sum_p > 100 else None
        print(f"  inset={inset}: pixels={sum_p:>6}  allowed->{r_allowed}  unrestricted->{r_full}")

    res = handwriting.read_letter_box(img, box, expected_set=allowed)
    print(f"  read_letter_box: text={res.text!r} conf={res.confidence:.4f} "
          f"source={res.source} needs_review={res.needs_review}")
    if res.raw_debug:
        for k, v in res.raw_debug.items():
            print(f"    {k}: {v}")
    print()
