"""Focused regressions for the handoff fixes (unittest, no model training)."""
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools/train"))
sys.path.insert(0, str(ROOT / "tools/eval"))


class CoreTests(unittest.TestCase):
    def test_group_split_is_reproducible_and_disjoint(self):
        from validation_split import split_validation
        rows = [{"instance_id": f"conv_{i}#p{j}"} for i in range(10) for j in range(3)]
        a, b = split_validation(rows)
        self.assertEqual((a, b), split_validation(rows))
        self.assertEqual(set(a) | set(b), set(range(len(rows))))
        self.assertFalse({rows[i]["instance_id"].split("#")[0] for i in a}
                         & {rows[i]["instance_id"].split("#")[0] for i in b})

    def test_legacy_windows_base_relocated(self):
        from model_paths import portable_base, resolve_base
        with tempfile.TemporaryDirectory() as d:
            base = Path(d) / "models/chinese-roberta-wwm-ext"
            base.mkdir(parents=True)
            (base / "config.json").write_text("{}", encoding="utf-8")
            old = r"C:\old\checkout\models/chinese-roberta-wwm-ext"
            self.assertEqual(resolve_base(d, old), str(base.resolve()))
            self.assertEqual(portable_base(d, base), "models/chinese-roberta-wwm-ext")
            with self.assertRaises(FileNotFoundError):
                resolve_base(d, old, "models/missing")

    def test_scorer_rejects_missing_duplicate_and_unknown_ids(self):
        from official_scorer import align_predictions
        gold = [{"instance_id": "a"}, {"instance_id": "b"}]
        with self.assertRaises(ValueError):
            align_predictions(gold, [{"instance_id": "a"}])
        with self.assertRaises(ValueError):
            align_predictions(gold, [{"instance_id": "a"}, {"instance_id": "a"}])
        with self.assertRaises(ValueError):
            align_predictions(gold, [{"instance_id": "unknown"}], allow_partial=True)
        preds, n = align_predictions(gold, [{"instance_id": "a"}], allow_partial=True)
        self.assertEqual((len(preds), n), (1, 2))
        preds, n = align_predictions(gold, [{"instance_id": "a"}], limit=1)
        self.assertEqual((len(preds), n), (1, 1))

    def test_downloader_atomic_failure_and_short_json(self):
        import download_ms
        class Response(io.BytesIO):
            headers = {"Content-Length": "2"}
        with tempfile.TemporaryDirectory() as d:
            path = str(Path(d) / "config.json")
            with patch.object(download_ms.urllib.request, "urlopen", return_value=Response(b"{}")):
                self.assertEqual(download_ms.download("repo", "config.json", path, retries=1), 2)
            with patch.object(download_ms.urllib.request, "urlopen", side_effect=OSError("test failure")), patch.object(download_ms.time, "sleep"):
                self.assertEqual(download_ms.download("repo", "config.json", path, retries=1), -1)
            self.assertEqual(Path(path).read_text(), "{}")
            with patch.object(download_ms, "download", return_value=-1), patch.object(sys, "argv", ["download_ms", "repo", d, "required"]):
                self.assertEqual(download_ms.main(), 1)


class AppTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            from fastapi.testclient import TestClient
        except ImportError:
            raise unittest.SkipTest("Install requirements.txt for service tests")
        cls.temp = tempfile.TemporaryDirectory()
        os.environ["STATE_DIR"] = cls.temp.name
        os.environ["LLM_BACKEND"] = "none"
        from app import server
        cls.server = server
        cls.client = TestClient(server.app)

    @classmethod
    def tearDownClass(cls):
        cls.client.close()
        cls.temp.cleanup()

    def setUp(self):
        self.server.SESSIONS.clear()
        self.server.SESSION_USERS.clear()
        self.server.store.users.clear()
        self.prompts = []
        async def capture(messages, **kwargs):
            self.prompts.append(messages)
            return "test response"
        self.mock = patch.object(self.server.llm, "chat", capture)
        self.mock.start()
        self.addCleanup(self.mock.stop)

    def test_anonymous_requests_are_stateless(self):
        for text in ["private marker alpha", "unrelated beta"]:
            r = self.client.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": text}]})
            self.assertEqual(r.status_code, 200)
        self.assertFalse(any("private marker alpha" in m["content"] for m in self.prompts[-1]))
        self.assertFalse(self.server.SESSIONS)
        self.assertFalse(self.server.store.users)

    def test_empty_history_overrides_session_and_system_is_filtered(self):
        self.client.post("/chat", json={"session_id": "s", "message": "old marker"})
        self.client.post("/chat", json={"session_id": "s", "message": "new", "history": []})
        history_messages = [m for m in self.prompts[-1] if m["role"] != "system"]
        self.assertEqual(history_messages, [{"role": "user", "content": "new"}])
        self.client.post("/chat", json={"session_id": "s", "message": "new", "history": [{"role": "system", "content": "untrusted marker"}]})
        self.assertFalse(any(m["content"] == "untrusted marker" for m in self.prompts[-1]))

    def test_reset_clears_custom_user_and_associated_sessions(self):
        for sid in ["s1", "s2"]:
            self.client.post("/chat", json={"session_id": sid, "user_id": "u", "message": "test"})
        self.client.post("/session/reset", json={"session_id": "s1"})
        self.assertNotIn("u", self.server.store.users)
        self.assertFalse(self.server.SESSIONS)
        self.assertFalse(self.server.SESSION_USERS)

    def test_crisis_regressions(self):
        from app.safety import assess_risk
        positives = ["我已经吞下一整瓶药，头很晕", "今晚准备从天台跳下去", "不想活了", "我想自杀", "我不想自杀，但已经吞下一整瓶药"]
        negatives = ["医生给我开了安眠药", "我不想自杀，只是有点焦虑", "我不准备从天台跳下去", "我没有吞下一整瓶药", "请不要自杀", "他终于解脱了"]
        for text in positives:
            self.assertEqual(assess_risk(text)[0], 3, text)
        for text in negatives:
            self.assertLess(assess_risk(text)[0], 3, text)
        r = self.client.post("/chat", json={"message": positives[0]})
        self.assertEqual(r.json()["risk"], 3)
        self.assertFalse(self.prompts)


class MLTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import torch
            import transformers
        except ImportError:
            raise unittest.SkipTest("Install training dependencies for ML tests")
        cls.torch = torch

    def test_exact_profile_per_sample(self):
        import torch
        from labels import PROFILE_GROUPS
        from train_multitask import evaluate
        class FakeModel:
            def __init__(self, pred):
                self.pred = pred
            def eval(self):
                return self
            def __call__(self, input_ids, attention_mask):
                return {"emotion": torch.tensor([[8.] + [-8.] * 15] * len(input_ids)),
                        **{k: torch.where(v.bool(), 8., -8.) for k, v in self.pred.items()}}
        gold = {k: torch.zeros((4, len(v))) for k, v in PROFILE_GROUPS.items()}
        gold["personality_traits"][0, 0] = 1
        pred = {k: v.clone() for k, v in gold.items()}
        batch = {"input_ids": torch.zeros((4, 2), dtype=torch.long), "attention_mask": torch.ones((4, 2)), "emotion": torch.zeros(4, dtype=torch.long), **gold}
        self.assertEqual(evaluate(FakeModel(pred), [batch], "cpu")["profile_exact"], 1.)
        pred["interests"][2, 0] = 1
        self.assertEqual(evaluate(FakeModel(pred), [batch], "cpu")["profile_exact"], .75)

    def test_latest_user_and_generation_marker_survive(self):
        from transformers import AutoTokenizer
        from infer_sft import BASE_SYSTEM, conditioned_system
        tok = AutoTokenizer.from_pretrained(str(ROOT / "models/t3_lora/smoke"), local_files_only=True)
        tok.truncation_side = "left"
        prompt = tok.apply_chat_template([
            {"role": "system", "content": BASE_SYSTEM},
            {"role": "user", "content": "old context " * 2000},
            {"role": "assistant", "content": "old reply " * 100},
            {"role": "user", "content": "LATEST_MARKER_123"}], tokenize=False, add_generation_prompt=True)
        ids = tok(prompt, truncation=True, max_length=2048, add_special_tokens=False)["input_ids"]
        text = tok.decode(ids, skip_special_tokens=False)
        self.assertIn("LATEST_MARKER_123", text)
        self.assertTrue(text.endswith("<|im_start|>assistant\n"))
        self.assertIn("anxiety", conditioned_system("anxiety", {"style": []}))


if __name__ == "__main__":
    unittest.main()
