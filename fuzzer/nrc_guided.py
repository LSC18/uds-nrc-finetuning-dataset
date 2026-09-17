"""NRC parsing, repair rules, and the minimal feedback loop."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

from ecu import Nrc, VirtualEcu


NRC_NAMES = {
    Nrc.SERVICE_NOT_SUPPORTED: "Service Not Supported",
    Nrc.SUBFUNCTION_NOT_SUPPORTED: "Sub-function Not Supported",
    Nrc.INCORRECT_MESSAGE_LENGTH_OR_INVALID_FORMAT: (
        "Incorrect Message Length Or Invalid Format"
    ),
    Nrc.REQUEST_SEQUENCE_ERROR: "Request Sequence Error",
    Nrc.REQUEST_OUT_OF_RANGE: "Request Out Of Range",
    Nrc.SECURITY_ACCESS_DENIED: "Security Access Denied",
    Nrc.INVALID_KEY: "Invalid Key",
    Nrc.SERVICE_NOT_SUPPORTED_IN_ACTIVE_SESSION: (
        "Service Not Supported In Active Session"
    ),
}


def format_hex(data: bytes) -> str:
    return data.hex(" ").upper()


@dataclass(frozen=True)
class ParsedResponse:
    raw: bytes
    positive: bool
    request_sid: int | None = None
    nrc: Nrc | None = None


def parse_response(response: bytes) -> ParsedResponse:
    """Parse the UDS positive/negative response shape used by the PoC."""
    if not response:
        raise ValueError("UDS response is empty")

    if response[0] != VirtualEcu.NEGATIVE_RESPONSE_SID:
        return ParsedResponse(raw=response, positive=True)

    if len(response) != 3:
        raise ValueError("Negative UDS response must be exactly three bytes")

    try:
        nrc = Nrc(response[2])
    except ValueError as exc:
        raise ValueError(f"Unsupported NRC: 0x{response[2]:02X}") from exc

    return ParsedResponse(
        raw=response,
        positive=False,
        request_sid=response[1],
        nrc=nrc,
    )


class RepairEngine:
    """Apply minimal, explicit NRC repair rules."""

    EXPECTED_LENGTH = {VirtualEcu.DIAGNOSTIC_SESSION_CONTROL: 2}
    DEFAULT_PAYLOAD = {VirtualEcu.DIAGNOSTIC_SESSION_CONTROL: bytes((0x01,))}

    def repair(self, request: bytes, parsed: ParsedResponse) -> bytes | None:
        if parsed.positive or parsed.nrc is None or parsed.request_sid is None:
            return None

        if parsed.nrc != Nrc.INCORRECT_MESSAGE_LENGTH_OR_INVALID_FORMAT:
            return None

        sid = parsed.request_sid
        expected_length = self.EXPECTED_LENGTH.get(sid)
        default_payload = self.DEFAULT_PAYLOAD.get(sid)
        if expected_length is None or default_payload is None:
            return None

        repaired = bytes((sid,)) + (request[1:] + default_payload)
        return repaired[:expected_length]


@dataclass(frozen=True)
class PocResult:
    success: bool
    attempts: int
    repairs: int
    final_request: bytes
    final_response: bytes


class JsonlTrace:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("", encoding="utf-8")

    def write(self, event: str, **fields: object) -> None:
        record = {
            "timestamp": datetime.now().astimezone().isoformat(timespec="milliseconds"),
            "event": event,
            **fields,
        }
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")


class NrcGuidedRunner:
    def __init__(
        self,
        ecu: VirtualEcu,
        repair_engine: RepairEngine,
        log_path: Path,
        emit: Callable[[str], None] = print,
    ) -> None:
        self.ecu = ecu
        self.repair_engine = repair_engine
        self.trace = JsonlTrace(log_path)
        self.emit = emit

    def run(self, initial_request: bytes, max_attempts: int = 3) -> PocResult:
        request = initial_request
        repairs = 0
        response = b""

        for attempt in range(1, max_attempts + 1):
            request_hex = format_hex(request)
            self.emit(f"[TX] {request_hex}")
            self.trace.write("tx", attempt=attempt, request=request_hex)

            response = self.ecu.handle_request(request)
            response_hex = format_hex(response)
            self.emit(f"[RX] {response_hex}")
            self.trace.write("rx", attempt=attempt, response=response_hex)

            parsed = parse_response(response)
            if parsed.positive:
                success = repairs > 0
                message = (
                    "NRC-guided repair succeeded"
                    if success
                    else "request was already valid; no repair performed"
                )
                self.emit(f"[RESULT] {message}")
                self.trace.write(
                    "result",
                    success=success,
                    attempts=attempt,
                    repairs=repairs,
                    message=message,
                )
                return PocResult(success, attempt, repairs, request, response)

            assert parsed.nrc is not None
            nrc_name = NRC_NAMES[parsed.nrc]
            self.emit(f"[NRC] 0x{int(parsed.nrc):02X} {nrc_name}")
            self.trace.write(
                "nrc",
                attempt=attempt,
                request_sid=f"0x{parsed.request_sid:02X}",
                nrc=f"0x{int(parsed.nrc):02X}",
                meaning=nrc_name,
            )

            repaired = self.repair_engine.repair(request, parsed)
            if repaired is None or repaired == request:
                message = "no applicable repair rule"
                self.emit(f"[RESULT] failed: {message}")
                self.trace.write(
                    "result",
                    success=False,
                    attempts=attempt,
                    repairs=repairs,
                    message=message,
                )
                return PocResult(False, attempt, repairs, request, response)

            repaired_hex = format_hex(repaired)
            self.emit(f"[REPAIR] {request_hex} -> {repaired_hex}")
            self.trace.write(
                "repair",
                attempt=attempt,
                nrc=f"0x{int(parsed.nrc):02X}",
                before=request_hex,
                after=repaired_hex,
            )
            request = repaired
            repairs += 1

        message = "maximum attempts reached"
        self.emit(f"[RESULT] failed: {message}")
        self.trace.write(
            "result",
            success=False,
            attempts=max_attempts,
            repairs=repairs,
            message=message,
        )
        return PocResult(False, max_attempts, repairs, request, response)
