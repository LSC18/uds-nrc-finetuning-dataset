"""Minimal transport-independent UDS virtual ECU.

This module intentionally models only the behavior needed for the first PoC.
It never communicates with a real vehicle, ECU, CAN interface, or network.
"""

from __future__ import annotations

from enum import IntEnum


class Nrc(IntEnum):
    SERVICE_NOT_SUPPORTED = 0x11
    SUBFUNCTION_NOT_SUPPORTED = 0x12
    INCORRECT_MESSAGE_LENGTH_OR_INVALID_FORMAT = 0x13
    REQUEST_SEQUENCE_ERROR = 0x24
    REQUEST_OUT_OF_RANGE = 0x31
    SECURITY_ACCESS_DENIED = 0x33
    INVALID_KEY = 0x35
    SERVICE_NOT_SUPPORTED_IN_ACTIVE_SESSION = 0x7E


class VirtualEcu:
    """Stateful UDS server model for isolated dataset and feedback-loop tests."""

    DIAGNOSTIC_SESSION_CONTROL = 0x10
    ECU_RESET = 0x11
    READ_DATA_BY_IDENTIFIER = 0x22
    SECURITY_ACCESS = 0x27
    WRITE_DATA_BY_IDENTIFIER = 0x2E
    TESTER_PRESENT = 0x3E
    POSITIVE_RESPONSE_OFFSET = 0x40
    NEGATIVE_RESPONSE_SID = 0x7F
    VIN_DID = 0xF190
    CONFIG_DID = 0xF1A0
    SECURITY_SEED = bytes((0x12, 0x34))
    SECURITY_KEY = bytes((0xBE, 0xEF))

    def __init__(self) -> None:
        self.active_session = 0x01
        self.supported_sessions = {0x01, 0x03}
        self.security_unlocked = False
        self.seed_requested = False
        self.config_value = 0x00

    @classmethod
    def _negative_response(cls, request_sid: int, nrc: Nrc) -> bytes:
        return bytes((cls.NEGATIVE_RESPONSE_SID, request_sid, int(nrc)))

    def handle_request(self, request: bytes) -> bytes:
        """Validate one UDS request and return a UDS-formatted response."""
        if not request:
            raise ValueError("UDS request must contain at least one SID byte")

        sid = request[0]

        handlers = {
            self.DIAGNOSTIC_SESSION_CONTROL: self._diagnostic_session_control,
            self.ECU_RESET: self._ecu_reset,
            self.READ_DATA_BY_IDENTIFIER: self._read_data_by_identifier,
            self.SECURITY_ACCESS: self._security_access,
            self.WRITE_DATA_BY_IDENTIFIER: self._write_data_by_identifier,
            self.TESTER_PRESENT: self._tester_present,
        }
        handler = handlers.get(sid)
        if handler is None:
            return self._negative_response(sid, Nrc.SERVICE_NOT_SUPPORTED)
        return handler(request)

    def state_name(self) -> str:
        session = "extended_session" if self.active_session == 0x03 else "default_session"
        security = "unlocked" if self.security_unlocked else "locked"
        return f"{session}:{security}"

    def _diagnostic_session_control(self, request: bytes) -> bytes:
        sid = request[0]
        if len(request) != 2:
            return self._negative_response(
                sid, Nrc.INCORRECT_MESSAGE_LENGTH_OR_INVALID_FORMAT
            )
        subfunction = request[1] & 0x7F
        if subfunction not in self.supported_sessions:
            return self._negative_response(sid, Nrc.SUBFUNCTION_NOT_SUPPORTED)
        self.active_session = subfunction
        self.security_unlocked = False
        self.seed_requested = False
        return bytes((sid + self.POSITIVE_RESPONSE_OFFSET, request[1]))

    def _ecu_reset(self, request: bytes) -> bytes:
        sid = request[0]
        if len(request) != 2:
            return self._negative_response(sid, Nrc.INCORRECT_MESSAGE_LENGTH_OR_INVALID_FORMAT)
        if (request[1] & 0x7F) != 0x01:
            return self._negative_response(sid, Nrc.SUBFUNCTION_NOT_SUPPORTED)
        response = bytes((sid + self.POSITIVE_RESPONSE_OFFSET, request[1]))
        self.__init__()
        return response

    def _read_data_by_identifier(self, request: bytes) -> bytes:
        sid = request[0]
        if len(request) != 3:
            return self._negative_response(sid, Nrc.INCORRECT_MESSAGE_LENGTH_OR_INVALID_FORMAT)
        did = int.from_bytes(request[1:3], "big")
        if did == self.VIN_DID:
            return bytes((sid + self.POSITIVE_RESPONSE_OFFSET, *request[1:3])) + b"VIRTUAL-ECU-00001"
        if did == self.CONFIG_DID:
            return bytes((sid + self.POSITIVE_RESPONSE_OFFSET, *request[1:3], self.config_value))
        return self._negative_response(sid, Nrc.REQUEST_OUT_OF_RANGE)

    def _security_access(self, request: bytes) -> bytes:
        sid = request[0]
        if self.active_session != 0x03:
            return self._negative_response(sid, Nrc.SERVICE_NOT_SUPPORTED_IN_ACTIVE_SESSION)
        if len(request) < 2:
            return self._negative_response(sid, Nrc.INCORRECT_MESSAGE_LENGTH_OR_INVALID_FORMAT)
        subfunction = request[1] & 0x7F
        if subfunction == 0x01:
            if len(request) != 2:
                return self._negative_response(sid, Nrc.INCORRECT_MESSAGE_LENGTH_OR_INVALID_FORMAT)
            self.seed_requested = True
            return bytes((sid + self.POSITIVE_RESPONSE_OFFSET, request[1])) + self.SECURITY_SEED
        if subfunction == 0x02:
            if len(request) != 4:
                return self._negative_response(sid, Nrc.INCORRECT_MESSAGE_LENGTH_OR_INVALID_FORMAT)
            if not self.seed_requested:
                return self._negative_response(sid, Nrc.REQUEST_SEQUENCE_ERROR)
            if request[2:] != self.SECURITY_KEY:
                return self._negative_response(sid, Nrc.INVALID_KEY)
            self.security_unlocked = True
            self.seed_requested = False
            return bytes((sid + self.POSITIVE_RESPONSE_OFFSET, request[1]))
        return self._negative_response(sid, Nrc.SUBFUNCTION_NOT_SUPPORTED)

    def _write_data_by_identifier(self, request: bytes) -> bytes:
        sid = request[0]
        if len(request) != 4:
            return self._negative_response(sid, Nrc.INCORRECT_MESSAGE_LENGTH_OR_INVALID_FORMAT)
        if self.active_session != 0x03:
            return self._negative_response(sid, Nrc.SERVICE_NOT_SUPPORTED_IN_ACTIVE_SESSION)
        if not self.security_unlocked:
            return self._negative_response(sid, Nrc.SECURITY_ACCESS_DENIED)
        did = int.from_bytes(request[1:3], "big")
        if did != self.CONFIG_DID:
            return self._negative_response(sid, Nrc.REQUEST_OUT_OF_RANGE)
        self.config_value = request[3]
        return bytes((sid + self.POSITIVE_RESPONSE_OFFSET, *request[1:3]))

    def _tester_present(self, request: bytes) -> bytes:
        sid = request[0]
        if len(request) != 2:
            return self._negative_response(sid, Nrc.INCORRECT_MESSAGE_LENGTH_OR_INVALID_FORMAT)
        if (request[1] & 0x7F) != 0x00:
            return self._negative_response(sid, Nrc.SUBFUNCTION_NOT_SUPPORTED)
        return bytes((sid + self.POSITIVE_RESPONSE_OFFSET, request[1]))
