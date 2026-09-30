import json
import os
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

from flask import Flask

from llm_assistant import (
    ProviderStore,
    LLMError,
    _dpapi_protect,
    _dpapi_unprotect,
    _dpapi_capability,
    _gemini_schema,
    _http_json,
    call_provider,
    extract_paper,
    normalize_protocol,
    normalize_report,
    register_llm_assistant,
    render_arxiv_tex,
)


class LLMAssistantTests(unittest.TestCase):
    def test_groq_uses_locked_openai_compatible_json_endpoint(self):
        config = {
            "provider": "groq", "model": "llama-3.3-70b-versatile",
            "base_url": "https://api.groq.com/openai/v1", "api_key": "secret-test-key",
        }
        response = {
            "choices": [{"message": {"content": '{"status":"connected"}'}}],
            "usage": {"total_tokens": 7},
        }
        with patch("llm_assistant._http_json", return_value=response) as mocked:
            result, usage = call_provider(
                config, "Follow the schema.", "Connection test.",
                {"type": "object"}, "connection_test", max_output_tokens=100,
            )
        url, headers, payload = mocked.call_args.args[:3]
        self.assertEqual(url, "https://api.groq.com/openai/v1/chat/completions")
        self.assertEqual(headers["Authorization"], "Bearer secret-test-key")
        self.assertEqual(payload["response_format"], {"type": "json_object"})
        self.assertIn("valid JSON object", payload["messages"][0]["content"])
        self.assertEqual(result["status"], "connected")
        self.assertEqual(usage["total_tokens"], 7)

    def test_cloud_origins_are_locked_and_custom_http_is_loopback_only(self):
        with tempfile.TemporaryDirectory() as temporary:
            store = ProviderStore(Path(temporary) / "credentials.json")
            for provider, official in {
                "openai": "https://api.openai.com",
                "anthropic": "https://api.anthropic.com",
                "gemini": "https://generativelanguage.googleapis.com",
                "groq": "https://api.groq.com/openai/v1",
            }.items():
                store.update(provider, "test-model", official, "test-key", persist=False)
                with self.assertRaises(LLMError):
                    store.update(provider, "test-model", "https://attacker.example", "test-key", persist=False)
                with self.assertRaises(LLMError):
                    store.update(provider, "test-model", official.replace("https://", "http://"), "test-key", persist=False)

            store.update("openai_compatible", "local-model", "http://127.0.0.1:11434/v1", "", persist=False)
            store.update("openai_compatible", "local-model", "http://[::1]:11434/v1", "", persist=False)
            store.update("openai_compatible", "remote-model", "https://models.example/v1", "custom-key", persist=False)
            for unsafe in [
                "http://192.168.1.10:11434/v1",
                "http://models.example/v1",
                "https://user:password@models.example/v1",
                "https://models.example/v1?redirect=attacker",
                "https://models.example/v1#fragment",
            ]:
                with self.assertRaises(LLMError, msg=unsafe):
                    store.update("openai_compatible", "model", unsafe, "key", persist=False)

    def test_legacy_unsafe_cloud_origin_fails_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "credentials.json"
            path.write_text(json.dumps({
                "active_provider": "openai",
                "providers": {"openai": {"model": "gpt-test", "base_url": "http://attacker.example"}},
            }), encoding="utf-8")
            public = ProviderStore(path).public()
            self.assertEqual(public["providers"]["openai"]["base_url"], "https://api.openai.com")

    def test_provider_redirect_is_not_followed_or_given_authorization(self):
        reached_target = []

        class TargetHandler(BaseHTTPRequestHandler):
            def do_GET(self):
                reached_target.append(self.headers.get("Authorization"))
                self.send_response(200)
                self.end_headers()

            def log_message(self, *_args):
                pass

        target = ThreadingHTTPServer(("127.0.0.1", 0), TargetHandler)

        class RedirectHandler(BaseHTTPRequestHandler):
            def do_POST(self):
                self.send_response(302)
                self.send_header("Location", f"http://127.0.0.1:{target.server_port}/capture")
                self.end_headers()

            def log_message(self, *_args):
                pass

        redirect = ThreadingHTTPServer(("127.0.0.1", 0), RedirectHandler)
        threads = [threading.Thread(target=server.serve_forever, daemon=True) for server in (target, redirect)]
        for thread in threads:
            thread.start()
        try:
            with self.assertRaisesRegex(LLMError, "HTTP 302"):
                _http_json(
                    f"http://127.0.0.1:{redirect.server_port}/start",
                    {"Authorization": "Bearer must-not-leak"}, {"test": True}, timeout=2,
                )
            self.assertEqual(reached_target, [])
        finally:
            redirect.shutdown()
            target.shutdown()
            redirect.server_close()
            target.server_close()

    def test_cross_site_requests_cannot_use_configured_provider(self):
        with tempfile.TemporaryDirectory() as temporary:
            (Path(temporary) / "geometry_lab").mkdir()
            app = Flask(__name__)
            register_llm_assistant(app, temporary)
            response = app.test_client().post(
                "/api/llm/test", json={},
                headers={"Origin": "https://malicious.example", "Sec-Fetch-Site": "cross-site"},
            )
            self.assertEqual(response.status_code, 403)

    @unittest.skipUnless(os.name == "nt", "Windows DPAPI test")
    def test_windows_dpapi_round_trip(self):
        available, reason = _dpapi_capability()
        if available:
            protected = _dpapi_protect("local-secret")
            self.assertNotIn("local-secret", protected)
            self.assertEqual(_dpapi_unprotect(protected), "local-secret")
        else:
            self.assertIn("DPAPI", reason)

    def test_session_key_never_appears_in_persisted_settings(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "credentials.json"
            store = ProviderStore(path)
            public = store.update(
                "openai", "gpt-test", "https://api.openai.com",
                "secret-test-key", persist=False,
            )
            persisted = path.read_text(encoding="utf-8")
            self.assertNotIn("secret-test-key", persisted)
            self.assertNotIn("secret-test-key", json.dumps(public))
            self.assertEqual(store.resolve("openai")["api_key"], "secret-test-key")

    def test_protocol_output_is_normalized_and_review_gated(self):
        raw = {
            "paper": {"title": "A & B", "citation": "Authors", "url": "", "summary": "Summary", "hypothesis": "H"},
            "dataset": {
                "source": "unknown-private-data", "dataset_type": "unknown",
                "factor_types": ["linear"], "factor_count": 999, "latent_dim": 2,
                "sparsity": 8, "dataset_size": 5, "observation_dim": 1,
                "noise": -2, "train_split": 2, "validation_split": 2,
                "seed": 7, "mixing": "orthogonal", "degeneracy": 0, "notes": "",
            },
            "model": {
                "model_type": "arbitrary_python", "latent_dim": 3, "beta": 4,
                "custom_model": json.dumps({"encoder_layers": [64, 32], "evil": "ignored"}), "notes": "",
            },
            "training": {
                "epochs": 0, "training_steps": 50, "learning_rate": 9,
                "batch_size": 1, "optimizer": "bad", "adam_beta1": .9,
                "adam_beta2": .999, "optimizer_epsilon": 1e-8,
                "weight_decay": 0, "reconstruction_loss": "bad",
                "hidden_dim": 128, "encoder_depth": 2, "decoder_depth": 2,
                "encoder_activation": "relu", "decoder_activation": "relu",
                "bias": True, "tc_weight": 6, "seed": 11,
                "paper_reproduction": True, "notes": "",
            },
            "analysis_plan": ["representation", "execute_shell"],
            "seed_plan": {"training_seeds": [1, 2], "metric_seed": 0, "aggregation": "mean", "notes": ""},
            "reproduction_limits": ["Batch size absent"], "unresolved": ["batch_size"],
            "evidence": [{"field": "beta", "value": "4", "source_location": "Table 1", "confidence": .9, "status": "reported"}],
        }
        draft = normalize_protocol(raw, {"sha256": "abc", "url": ""})
        self.assertTrue(draft["review_required"])
        self.assertEqual(draft["dataset"]["source"], "uploaded")
        self.assertEqual(draft["dataset"]["factor_count"], 32)
        self.assertEqual(draft["dataset"]["sparsity"], 1.0)
        self.assertEqual(draft["model"]["model_type"], "custom_vae")
        self.assertEqual(draft["model"]["custom_model"]["encoder_layers"], [64, 32])
        self.assertNotIn("evil", draft["model"]["custom_model"])
        self.assertEqual(draft["training"]["learning_rate"], 1.0)
        self.assertEqual(draft["analysis_plan"], ["representation"])

    def test_paper_text_limit_and_source_hash(self):
        text = "Methods and results. " * 50
        extracted, source = extract_paper(text=text)
        self.assertEqual(extracted, text.strip())
        self.assertEqual(source["kind"], "text")
        self.assertGreaterEqual(source["characters_used"], 500)

    def test_gemini_nullable_schema_translation(self):
        schema = _gemini_schema({
            "type": "object", "additionalProperties": False,
            "properties": {"seed": {"type": ["integer", "null"]}},
        })
        self.assertEqual(schema["properties"]["seed"]["type"], "integer")
        self.assertTrue(schema["properties"]["seed"]["nullable"])
        self.assertNotIn("additionalProperties", schema)

    def test_arxiv_source_escapes_model_text(self):
        report = normalize_report({
            "title": "A&B_Study", "abstract": "100% measured", "introduction": "Intro",
            "state_of_the_art": "Prior", "theory": "Theory", "methodology": "Method",
            "results": "Result", "discussion": "Discussion", "limitations": "Limits",
            "perspectives": "Future", "conclusion": "Conclusion", "key_findings": ["x < y"],
            "data_caveats": ["None claimed"], "references": [{"citation": "Author #1", "url": ""}],
        })
        tex = render_arxiv_tex(report, {"run_count": 2})
        self.assertIn(r"A\&B\_Study", tex)
        self.assertIn(r"100\% measured", tex)
        self.assertIn(r"\documentclass", tex)


if __name__ == "__main__":
    unittest.main()
