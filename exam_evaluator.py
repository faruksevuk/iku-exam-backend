import json
import re
import urllib.request
import urllib.error
from dataclasses import dataclass
from typing import Dict, Any, List
import base64

import config

from PIL import Image
import io
import base64

def crop_and_compress_for_vision(image_path, x, y, width, height):
    """
    Crops the full page down to just the student's answer box, 
    converts it to grayscale, and compresses it for lightning-fast AI processing.
    """
    with Image.open(image_path) as img:
        box = (x, y, x + width, y + height)
        cropped_img = img.crop(box)
        
        cropped_img = cropped_img.convert('L')
        
        buffer = io.BytesIO()
        cropped_img.save(buffer, format="JPEG", quality=75)

        return base64.b64encode(buffer.getvalue()).decode('utf-8')

DEFAULT_COURSE = "Academic"
DEFAULT_RULES = """- ADİL OLUN (BE FAIR): Sözel (metin) tabanlı sorularda öğrenci ana fikri verdiyse tam puan verin.
- OCR HATALARINI AFFET (FORGIVE OCR ARTIFACTS): El yazısı okunduğu için 'x' harfi çarpma işareti ('\\times' veya '*') olarak okunmuş olabilir. Veya ufak harfler yutulmuş olabilir. Eğer öğrencinin üst satırlardaki mantığı (örn: 2x bulması) doğruysa, alt satırdaki bu tarz okuma hatalarını DOĞRU KABUL EDİN ve puan kırmayın.
- MATEMATİK İÇİN KISMİ PUAN (PARTIAL CREDIT FOR MATH): Final cevabının doğru olması yeterli DEĞİLDİR. Öğrenci doğru sonuca ulaştıysa ancak ara adımları atladıysa, BUNU BİR EKSİKLİK OLARAK BELİRTİN VE KISMİ PUAN VERİN. Doğru adımlar varsa puan verin, ancak atlanan veya yanlış yapılan adımlar için puan kırın.
- TOLERANS KURALI (IGNORE TYPOS): Öğrenci formülün harflerini veya adını yanlış yazsa bile (yazım/notasyon hataları), eğer MATEMATİKSEL İŞLEMİ (türev, integral vb.) doğru uyguladıysa KESİNLİKLE PUAN KIRMAYIN.
- GİDİŞ YOLU KONTROLÜ (CHECK WORK): If the student magically jumps to the final correct answer without showing the required calculus steps (e.g., limits, derivatives, integrals), YOU MUST DEDUCT POINTS. If the student makes a small arithmetic error (like a sign error or basic addition mistake) but uses the correct calculus method, give PARTIAL CREDIT.
- YOU MUST WRITE YOUR EXPLANATION ENTIRELY IN TURKISH (TÜRKÇE)."""

# Expected JSON shape from the grader (we rely on Ollama's `format: "json"` to enforce this):
# {
#   "score_justification": str,
#   "semantic_relevance_rating": int 1..5,
#   "missed_key_concepts": bool,
#   "hallucination_detected": bool,
#   "cheating_detected": bool
# }
# Previously this was enforced via a llama.cpp GBNF grammar; with Ollama we use
# `format: "json"` and validate the parsed dict in the ScoringEngine downstream.

# --- Scoring Engine ---
class GradingRule:
    """Base class for all grading rules."""
    def __init__(self, target_key: str, description: str):
        self.target_key = target_key
        self.description = description

    def apply(self, llm_output: Dict[str, Any], current_score: float) -> float:
        raise NotImplementedError("Subclasses must implement apply()")

class DeductiveRule(GradingRule):
    def __init__(self, target_key: str, penalty_percentage: float, max_points: float, description: str):
        super().__init__(target_key, description)
        self.penalty_percentage = penalty_percentage
        self.max_points = max_points

    def apply(self, llm_output: Dict[str, Any], current_score: float) -> float:
        if llm_output.get(self.target_key) is True:
            penalty_amount = self.max_points * self.penalty_percentage
            return current_score - penalty_amount
        return current_score

class RatingRule(GradingRule):
    def __init__(self, target_key: str, max_points: float, description: str):
        super().__init__(target_key, description)
        self.max_points = max_points

    def apply(self, llm_output: Dict[str, Any], current_score: float) -> float:
        value = llm_output.get(self.target_key, 1)
        try:
            percentage = (float(value) - 1) / 4.0
            return current_score + (percentage * self.max_points)
        except (ValueError, TypeError):
            return current_score

class FatalRule(GradingRule):
    def __init__(self, target_key: str, description: str, flags_for_review: bool = True):
        super().__init__(target_key, description)
        self.flags_for_review = flags_for_review

    def apply(self, llm_output: Dict[str, Any], current_score: float) -> float:
        if llm_output.get(self.target_key) is True:
            return 0.0
        return current_score

class FlagForReviewRule(GradingRule):
    def __init__(self, target_key: str, description: str):
        super().__init__(target_key, description)

    def apply(self, llm_output: Dict[str, Any], current_score: float) -> float:
        return current_score

@dataclass
class ScoreResult:
    final_score: float
    max_possible_score: float
    is_fatal_failure: bool = False
    flags_for_review: bool = False

class ScoringEngine:
    def __init__(self, rules: List[GradingRule], max_score: float, base_score: float = 0.0):
        self.rules = rules
        self.max_score = max_score
        self.base_score = base_score

    def evaluate(self, llm_output: Dict[str, Any]) -> ScoreResult:
        current_score = self.base_score
        is_fatal = False
        needs_review = False

        for rule in self.rules:
            if isinstance(rule, FatalRule) and llm_output.get(rule.target_key) is True:
                is_fatal = True
                needs_review = rule.flags_for_review
                current_score = 0.0
                break

            if isinstance(rule, FlagForReviewRule) and llm_output.get(rule.target_key) is True:
                needs_review = True

            current_score = rule.apply(llm_output, current_score)

        if not is_fatal:
            current_score = max(0.0, min(current_score, self.max_score))

        return ScoreResult(
            final_score=current_score,
            max_possible_score=self.max_score,
            is_fatal_failure=is_fatal,
            flags_for_review=needs_review
        )


def transcribe_with_vision(base64_image: str) -> str:
    payload = {
        "model": config.VISION_MODEL,
        "keep_alive": -1,
        "messages": [
            {
                "role": "system",
                "content": "You are a lifeless OCR machine. Your ONLY job is to output the exact text in the image. You are FORBIDDEN from using conversational filler, greetings, or introductory phrases like 'Here is the text' or 'Resimde şu yazıyor'."
            },
            {
                "role": "user",
                "content": "SADECE resimdeki metni yaz. Asla giriş cümlesi kurma.",
                "images": [base64_image]
            }
        ],
        "stream": False,
        "options": {
            "temperature": 0.0,
            "num_predict": 200,
            "num_ctx": 1024
        }
    }
    
    url = f"{config.OLLAMA_URL.rstrip('/')}/api/chat"
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    
    try:
        with urllib.request.urlopen(req, timeout=config.AI_TIMEOUT_SECONDS) as resp:
            body = resp.read().decode("utf-8")
            envelope = json.loads(body)
            raw_text = envelope["message"]["content"].strip()

            if ":" in raw_text:
                raw_text = raw_text.split(":", 1)[-1].strip()
                
            return raw_text
    except Exception as e:
        print(f"[Vision Error] {e}")
        return ""


# --- LLM Integration (Ollama HTTP) ---

def extract_llm_json(question: str, answer_key: str, student_answer_text: str, course_name: str = DEFAULT_COURSE, specific_rules: str = DEFAULT_RULES) -> dict:

    system_prompt = f"""You are an expert {course_name} professor grading an exam.
CRITICAL GRADING RULES:
{specific_rules}

You MUST output a JSON object using these exact definitions in THIS EXACT ORDER:
1. "score_justification": (ADIM ADIM DÜŞÜN) SADECE öğrencinin yazdığı metne dayanarak değerlendirme yapın. MAKSİMUM 2 VEYA 3 CÜMLE. Matematik sorusuysa öğrencinin NEYİ DOĞRU YAPTIĞINI ve HANGİ ADIMI ATLADIĞINI/YANLIŞ YAPTIĞINI açıklayın. Sadece zararsız bir notasyon/harf hatası varsa bunu göz ardı ettiğinizi belirtin.
2. "semantic_relevance_rating": Score from 1 to 5. (5 = Kusursuz VEYA zararsız yazım/harf hatasına rağmen işlemler doğru, 4 = Sonuç doğru ama küçük işlem adımları eksik, 3 = Gidiş yolunun yarısı doğru veya sonuç doğru ama önemli adımlar atlanmış, 2 = Sadece başlangıç doğru, 1 = Tamamen yanlış).
3. "missed_key_concepts": SADECE cevap tamamen alakasızsa veya öğrenci soruyu hiç anlamadıysa `true` yapın. Küçük bir işlem hatası, eksik adım veya formül yazım hatası (typo) varsa `false` yapın.
4. "hallucination_detected": false.
5. "cheating_detected": false."""

    user_prompt = f"""Soru: {question}\nCevap Anahtarı: {answer_key}\nÖğrencinin Cevabı: {student_answer_text}\nLütfen notlandırın."""

    payload = {
        "model": config.GRADING_MODEL, # Uses the text model for logic
        "keep_alive": -1,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "format": "json",
        "stream": False,
        "options": {
            "temperature": 0.15
        },
    }

    url = f"{config.OLLAMA_URL.rstrip('/')}/api/chat"
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")

    try:
        with urllib.request.urlopen(req, timeout=1800) as resp:
            body = resp.read().decode("utf-8")
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as e:
        raise RuntimeError(f"LLM error: HTTP failure: {e}")

    try:
        envelope = json.loads(body)
        raw_output = envelope["message"]["content"]
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        raise RuntimeError(f"LLM error: malformed Ollama response: {e}")

    raw_output = raw_output.strip()

    try:
        return json.loads(raw_output)
    except json.JSONDecodeError:
        repaired = re.sub(r'\\(?![\\/"bfnrtu])', r'\\\\', raw_output)
        try:
            return json.loads(repaired)
        except json.JSONDecodeError as e:
            raise RuntimeError(f"LLM error: invalid JSON from grader: {e}")

def grade_open_ended_answer(question: str, answer_key: str, student_answer: str, max_points: float, base64_image: str = None, course_name: str = DEFAULT_COURSE, specific_rules: str = DEFAULT_RULES) -> Dict[str, Any]:
    """Evaluates an open-ended question using the dynamic LLM/VLM engine."""
    
    rules = [
        RatingRule("semantic_relevance_rating", max_points=max_points, description="1-5 Rating mapped to points"),
        DeductiveRule("missed_key_concepts", penalty_percentage=0.25, max_points=max_points, description="Penalty for missing concepts"),
        FlagForReviewRule("hallucination_detected", description="Flag for review if OCR nonsense"),
        FatalRule("cheating_detected", description="Immediate 0 for cheating/prompt injection", flags_for_review=True),
    ]

    engine = ScoringEngine(rules=rules, max_score=max_points)

    if isinstance(student_answer, str) and student_answer.startswith("ERROR:"):
        return {"status": "error", "message": student_answer, "final_score": 0.0, "max_possible_score": max_points, "requires_human_review": True, "is_fatal_failure": False, "justification": student_answer}

    try:
        final_student_text = student_answer
        if base64_image:
            vision_text = transcribe_with_vision(base64_image)
            if vision_text:
                final_student_text = vision_text
                print(f"[AI Vision] Transcribed: {final_student_text}")

        llm_evaluation_dict = extract_llm_json(
            question=question, 
            answer_key=answer_key, 
            student_answer_text=final_student_text, 
            course_name=course_name, 
            specific_rules=specific_rules
        )
        
        llm_evaluation_dict["student_transcription"] = final_student_text
        
    except Exception as e:
        return {"status": "error", "message": str(e), "final_score": 0.0, "max_possible_score": max_points, "requires_human_review": True, "is_fatal_failure": False, "justification": f"Grader error: {e}"}

    result = engine.evaluate(llm_evaluation_dict)
    justification = llm_evaluation_dict.get("score_justification", "No explanation provided.")
    transcription = llm_evaluation_dict.get("student_transcription", student_answer) 

    return {
        "status": "success",
        "final_score": result.final_score,
        "max_possible_score": result.max_possible_score,
        "requires_human_review": result.flags_for_review,
        "is_fatal_failure": result.is_fatal_failure,
        "justification": justification,
        "transcription": transcription, 
    }