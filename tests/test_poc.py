from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from ecu import Nrc, VirtualEcu
from fuzzer import NrcGuidedRunner, RepairEngine, parse_response


class VirtualEcuTests(unittest.TestCase):
    def test_short_session_request_returns_nrc_13(self) -> None:
        ecu = VirtualEcu()
        self.assertEqual(ecu.handle_request(bytes.fromhex("10")), bytes.fromhex("7F 10 13"))

    def test_valid_default_session_request_is_positive(self) -> None:
        ecu = VirtualEcu()
        self.assertEqual(
            ecu.handle_request(bytes.fromhex("10 01")), bytes.fromhex("50 01")
        )

    def test_security_access_requires_extended_session(self) -> None:
        ecu = VirtualEcu()
        self.assertEqual(
            ecu.handle_request(bytes.fromhex("27 01")), bytes.fromhex("7F 27 7E")
        )

    def test_unlock_then_write_and_read_config_did(self) -> None:
        ecu = VirtualEcu()
        requests = ("10 03", "27 01", "27 02 BE EF", "2E F1 A0 5A")
        responses = [ecu.handle_request(bytes.fromhex(request)) for request in requests]
        self.assertEqual(responses[-1], bytes.fromhex("6E F1 A0"))
        self.assertEqual(
            ecu.handle_request(bytes.fromhex("22 F1 A0")),
            bytes.fromhex("62 F1 A0 5A"),
        )
        self.assertEqual(ecu.state_name(), "extended_session:unlocked")

    def test_reset_restores_default_locked_state(self) -> None:
        ecu = VirtualEcu()
        for request in ("10 03", "27 01", "27 02 BE EF"):
            ecu.handle_request(bytes.fromhex(request))
        self.assertEqual(ecu.handle_request(bytes.fromhex("11 01")), bytes.fromhex("51 01"))
        self.assertEqual(ecu.state_name(), "default_session:locked")


class NrcGuidedTests(unittest.TestCase):
    def test_parser_extracts_nrc(self) -> None:
        parsed = parse_response(bytes.fromhex("7F 10 13"))
        self.assertFalse(parsed.positive)
        self.assertEqual(parsed.request_sid, 0x10)
        self.assertEqual(parsed.nrc, Nrc.INCORRECT_MESSAGE_LENGTH_OR_INVALID_FORMAT)

    def test_repair_engine_adds_default_subfunction(self) -> None:
        parsed = parse_response(bytes.fromhex("7F 10 13"))
        repaired = RepairEngine().repair(bytes.fromhex("10"), parsed)
        self.assertEqual(repaired, bytes.fromhex("10 01"))

    def test_full_feedback_loop_succeeds_and_logs(self) -> None:
        output: list[str] = []
        with tempfile.TemporaryDirectory() as directory:
            log_path = Path(directory) / "trace.jsonl"
            runner = NrcGuidedRunner(
                VirtualEcu(), RepairEngine(), log_path, emit=output.append
            )
            result = runner.run(bytes.fromhex("10"))

            self.assertTrue(result.success)
            self.assertEqual(result.attempts, 2)
            self.assertEqual(result.repairs, 1)
            self.assertEqual(result.final_response, bytes.fromhex("50 01"))
            self.assertTrue(log_path.exists())

        self.assertEqual(
            output,
            [
                "[TX] 10",
                "[RX] 7F 10 13",
                "[NRC] 0x13 Incorrect Message Length Or Invalid Format",
                "[REPAIR] 10 -> 10 01",
                "[TX] 10 01",
                "[RX] 50 01",
                "[RESULT] NRC-guided repair succeeded",
            ],
        )


if __name__ == "__main__":
    unittest.main()
