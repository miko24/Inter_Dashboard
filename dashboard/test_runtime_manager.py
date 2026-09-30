import unittest
from unittest.mock import patch

from flask import Flask

import runtime_manager
from runtime_manager import CONFIRMATION, _recommended_build, register_runtime_manager


class RuntimeManagerTests(unittest.TestCase):
    def setUp(self):
        self.app = Flask(__name__)
        register_runtime_manager(self.app)
        self.client = self.app.test_client()
        self.detected = {
            "available": True,
            "maximum_cuda": 13.3,
            "gpus": [{
                "name": "NVIDIA Test GPU", "driver_version": "610.47",
                "memory_mib": 8192, "compute_capability": "7.5",
            }],
        }

    def test_recommends_highest_allowlisted_compatible_build(self):
        self.assertEqual(_recommended_build(self.detected), "cu130")
        self.assertEqual(_recommended_build({"maximum_cuda": 12.5}), "cu124")
        self.assertIsNone(_recommended_build({"maximum_cuda": 11.0}))

    def test_status_reports_detected_gpu_and_fixed_builds(self):
        with patch.object(runtime_manager, "_nvidia_status", return_value=self.detected):
            response = self.client.get("/api/runtime/compute")
            cross_site = self.client.get(
                "/api/runtime/compute", headers={"Origin": "https://attacker.example"},
            )
        payload = response.get_json()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(payload["recommended_build"], "cu130")
        self.assertEqual(payload["nvidia"]["gpus"][0]["name"], "NVIDIA Test GPU")
        self.assertTrue(all(item["id"].startswith("cu") for item in payload["builds"]))
        self.assertEqual(cross_site.status_code, 403)

    def test_install_rejects_cross_origin_unapproved_build_and_missing_confirmation(self):
        with patch.object(runtime_manager, "_nvidia_status", return_value=self.detected):
            cross_site = self.client.post(
                "/api/runtime/pytorch-cuda/install",
                json={"build": "cu130", "confirmation": CONFIRMATION},
                headers={"Origin": "https://attacker.example"},
            )
            unapproved = self.client.post(
                "/api/runtime/pytorch-cuda/install",
                json={"build": "https://example.invalid/wheels", "confirmation": CONFIRMATION},
            )
            unconfirmed = self.client.post(
                "/api/runtime/pytorch-cuda/install", json={"build": "cu130", "confirmation": "yes"},
            )
        self.assertEqual(cross_site.status_code, 403)
        self.assertEqual(unapproved.status_code, 400)
        self.assertEqual(unconfirmed.status_code, 400)


if __name__ == "__main__":
    unittest.main()
