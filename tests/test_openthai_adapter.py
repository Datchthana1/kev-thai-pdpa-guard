"""Tests for src/kev_qlora/openthai_adapter.py.

Default run needs no model and no network:
    .venv/Scripts/python.exe -m pytest tests/test_openthai_adapter.py

Real model run (CPU, downloads iapp/OpenThai-SystemOne ~1.5 GB into HF_HOME):
    OPENTHAI_REAL=1 HF_HOME=D:/hf_cache CUDA_VISIBLE_DEVICES=-1 .venv/Scripts/python.exe -m pytest tests/test_openthai_adapter.py -k real
"""
import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from kev_qlora.openthai_adapter import OpenThaiSystem, parse_ollama_response  # noqa: E402

QUESTIONS = {
    "department": {
        "type": "choice",
        "instructions": "ทีมใดควรรับผิดชอบ",
        "criteria": {"billing": "การเงิน/ค่าบริการ", "technical": "ระบบใช้งานไม่ได้", "sales": None},
    },
    "frustration": {
        "type": "score",
        "instructions": "ลูกค้าหงุดหงิดแค่ไหน",
        "criteria": ["ใจเย็น", "หงุดหงิดแต่สุภาพ", "โกรธมาก"],
    },
    "refund_requested": {"type": "noul", "instructions": "ลูกค้าขอเงินคืนอย่างชัดเจนหรือไม่"},
}

# Shape of the Ollama /v1/systemone response, as documented on the model card (values rounded).
OLLAMA_CANNED = {
    "model": "iapp/openthai-systemone",
    "answers": {
        "department": {
            "type": "choice",
            "choice": "billing",
            "probabilities": {"billing": 0.9515, "technical": 0.0334, "sales": 0.0151},
            "confidence": 0.7959,
        },
        "frustration": {
            "type": "score",
            "score": 1.8771,
            "legend": {"0": "ใจเย็น", "1": "หงุดหงิดแต่สุภาพ", "2": "โกรธมาก"},
            "probabilities": {"0": 0.0216, "1": 0.0797, "2": 0.8987},
            "confidence": 0.6537,
        },
        "refund_requested": {"type": "noul", "noul": 0.9805},
    },
    "usage": {"input_tokens": 741, "output_tokens": 4},
}


def _assert_contract(answers, questions):
    assert set(answers) == set(questions)
    for qid, q in questions.items():
        a = answers[qid]
        assert a["type"] == q["type"]
        assert abs(sum(a["probs"].values()) - 1.0) < 1e-6, qid
        if q["type"] == "noul":
            assert set(a["probs"]) == {"false", "true"}
        elif q["type"] == "score":
            assert set(a["probs"]) == {str(i) for i in range(len(q["criteria"]))}
        else:
            assert set(a["probs"]) == set(q["criteria"])


# ---------------------------------------------------------------- parsing (no model, no network)
def test_parse_ollama_full_distributions():
    out = parse_ollama_response(OLLAMA_CANNED, QUESTIONS)
    _assert_contract(out, QUESTIONS)
    assert out["refund_requested"]["probs"]["true"] == pytest.approx(0.9805)
    assert out["refund_requested"]["probs"]["false"] == pytest.approx(0.0195)
    assert out["department"]["choice"] == "billing"
    assert out["department"]["probs"]["billing"] == pytest.approx(0.9515, abs=1e-4)
    assert out["frustration"]["probs"]["2"] == pytest.approx(0.8987, abs=1e-4)
    assert out["frustration"]["score"] == pytest.approx(1.8771)


def test_parse_ollama_choice_label_with_confidence_only():
    raw = {"answers": {"department": {"type": "choice", "choice": "technical", "confidence": 0.8}}}
    q = {"department": QUESTIONS["department"]}
    out = parse_ollama_response(raw, q)
    p = out["department"]["probs"]
    assert p["technical"] == pytest.approx(0.8)
    assert p["billing"] == pytest.approx(0.1)
    assert p["sales"] == pytest.approx(0.1)
    _assert_contract(out, q)


def test_parse_ollama_score_scalar_triangular_keeps_mean():
    raw = {"answers": {"frustration": {"type": "score", "score": 1.3}}}
    q = {"frustration": QUESTIONS["frustration"]}
    out = parse_ollama_response(raw, q)
    p = out["frustration"]["probs"]
    # triangular kernel of half-width 1 centred at 1.3: p(1)=0.7, p(2)=0.3, p(0)=0
    assert p["0"] == pytest.approx(0.0)
    assert p["1"] == pytest.approx(0.7)
    assert p["2"] == pytest.approx(0.3)
    expected = sum(int(k) * v for k, v in p.items())
    assert expected == pytest.approx(1.3)


def test_parse_ollama_score_scalar_clamped_at_edge():
    raw = {"answers": {"frustration": {"type": "score", "score": 7.5}}}
    q = {"frustration": QUESTIONS["frustration"]}
    out = parse_ollama_response(raw, q)
    assert out["frustration"]["probs"]["2"] == pytest.approx(1.0)
    _assert_contract(out, q)


def test_parse_ollama_missing_answer_raises():
    with pytest.raises(ValueError):
        parse_ollama_response({"answers": {}}, {"refund_requested": QUESTIONS["refund_requested"]})


def test_invalid_question_rejected_before_any_backend_call():
    s = OpenThaiSystem(backend="ollama", ollama_url="http://127.0.0.1:9")  # nothing listens here
    with pytest.raises(ValueError):
        s.predict("text", {"x": {"type": "ranking", "instructions": "?"}})


# ---------------------------------------------------------------- mocked HTTP (no model)
class _FakeOllama(BaseHTTPRequestHandler):
    seen = {}

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length)
        if self.path != "/v1/systemone":
            self.send_response(404)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        _FakeOllama.seen = {"path": self.path, "body": json.loads(raw.decode("utf-8"))}
        payload = json.dumps(OLLAMA_CANNED, ensure_ascii=False).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *args):  # keep test output quiet
        pass


@pytest.fixture
def fake_ollama():
    server = HTTPServer(("127.0.0.1", 0), _FakeOllama)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()


def test_ollama_backend_posts_contract_and_parses_response(fake_ollama):
    state = {"ticket": "ลูกค้าแจ้งว่าโดนหักเงินซ้ำสองครั้ง"}
    s = OpenThaiSystem(backend="ollama", ollama_url=fake_ollama, ollama_model="iapp/openthai-systemone:0.8b-q8_0")
    answers, latency = s.predict(state, QUESTIONS)

    assert _FakeOllama.seen["path"] == "/v1/systemone"
    body = _FakeOllama.seen["body"]
    assert body["model"] == "iapp/openthai-systemone:0.8b-q8_0"
    assert body["state"] == state
    assert body["questions"]["department"]["criteria"]["sales"] is None
    assert body["questions"]["frustration"]["criteria"][2] == "โกรธมาก"
    assert "criteria" not in body["questions"]["refund_requested"]

    _assert_contract(answers, QUESTIONS)
    assert answers["refund_requested"]["probs"]["true"] == pytest.approx(0.9805)
    assert latency >= 0.0
    assert s.backend == "ollama"


def test_ollama_http_error_is_reported(fake_ollama):
    s = OpenThaiSystem(backend="ollama", ollama_url=fake_ollama + "/nope")  # path not served -> 404
    with pytest.raises(RuntimeError, match="HTTP 404"):
        s.predict("x", {"refund_requested": QUESTIONS["refund_requested"]})


# ---------------------------------------------------------------- real model (opt-in)
@pytest.mark.skipif(os.environ.get("OPENTHAI_REAL") != "1", reason="set OPENTHAI_REAL=1 to run the real model")
def test_real_transformers_cpu_two_records():
    s = OpenThaiSystem(backend="transformers", device="cpu")
    with_id = "ลูกค้าชื่อสมชาย เลขบัตรประชาชน 1-1037-02345-67-1 ติดต่อเรื่องใบแจ้งหนี้ค่าบริการเดือนนี้"
    clean = "ลูกค้าชื่อสมหญิง ขอเปลี่ยนที่อยู่จัดส่งสินค้าเป็นกรุงเทพฯ ไม่มีข้อมูลส่วนบุคคลอ่อนไหว"
    questions = {
        "has_national_id": {"type": "noul", "instructions": "ข้อความนี้มีเลขบัตรประชาชนไทย 13 หลักหรือไม่"},
        "id_risk": {
            "type": "score",
            "instructions": "ระดับความเสี่ยงที่ข้อความมีข้อมูลระบุตัวตนส่วนบุคคล",
            "criteria": ["ไม่มีข้อมูลส่วนบุคคล", "มีข้อมูลระบุตัวตนเล็กน้อย", "มีข้อมูลระบุตัวตนปานกลาง",
                         "มีข้อมูลระบุตัวตนชัดเจน", "มีเลขบัตรประชาชนเต็มรูปแบบ"],
        },
        "topic": {
            "type": "choice",
            "instructions": "เรื่องหลักของข้อความนี้คืออะไร",
            "criteria": {"billing": "การเงิน/ค่าบริการ", "address": "ที่อยู่/การจัดส่ง",
                         "identity": "ข้อมูลตัวตนส่วนบุคคล", "other": None},
        },
    }
    a_id, lat_a = s.predict(with_id, questions)
    a_clean, lat_b = s.predict(clean, questions)
    _assert_contract(a_id, questions)
    _assert_contract(a_clean, questions)
    assert a_id["has_national_id"]["probs"]["true"] > 0.5
    assert a_clean["has_national_id"]["probs"]["true"] < 0.5
    assert a_id["id_risk"]["score"] > a_clean["id_risk"]["score"]
    assert lat_a > 0 and lat_b > 0
