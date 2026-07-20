"""Tests for ODIDParser.parse_system (ODID System message, type 4).

Guards the OperatorAltitudeGeo byte offset. The field lives at bytes 18-19 of
the System message; byte 17 is the UA Category/Class byte. A prior off-by-one
read the uint16 at offset 17, shifting the altitude's low byte into the high
position and producing garbage ~24 km operator altitudes (observed live on
kestrel1 DJI beacons).

ODID System message (type 4) layout, per ASTM F3411 / opendroneid-core-c:
  byte  0     : MessageType(4) | ProtoVersion(4)
  byte  1     : flags (OperatorLocationType | ClassificationType | reserved)
  bytes 2-5   : OperatorLatitude  (int32 LE, deg * 1e7)
  bytes 6-9   : OperatorLongitude (int32 LE, deg * 1e7)
  bytes 10-11 : AreaCount (uint16 LE)
  byte  12    : AreaRadius (uint8)
  bytes 13-14 : AreaCeiling (uint16 LE)
  bytes 15-16 : AreaFloor   (uint16 LE)
  byte  17    : CategoryEU(4) | ClassEU(4)
  bytes 18-19 : OperatorAltitudeGeo (uint16 LE, * 0.5 - 1000 m)
  bytes 20-23 : Timestamp (uint32 LE)
  byte  24    : Reserved
"""
import struct
import unittest

from sparrow_droneid.backend.droneid_engine import ODIDParser, ODID_MSG_SYSTEM
from sparrow_droneid.backend.models import DroneIDDevice


def _encode_alt(alt_m: float) -> int:
    """Inverse of the ODID altitude decode (value * 0.5 - 1000)."""
    return int(round((alt_m + 1000.0) / 0.5))


def _build_system_msg(
    op_lat_deg: float = 33.14097,
    op_lon_deg: float = -80.10749,
    operator_alt_m: float = 24.0,
    category_class_byte: int = 0x50,
    proto_version: int = 2,
) -> bytes:
    """Build a 25-byte ODID System message with a known operator altitude.

    category_class_byte is deliberately non-zero: if the parser regresses to
    reading the altitude at offset 17, this byte corrupts the result, so the
    altitude assertion fails.
    """
    msg = bytearray(25)
    msg[0] = (ODID_MSG_SYSTEM << 4) | (proto_version & 0x0F)
    msg[1] = 0x00
    struct.pack_into('<i', msg, 2, int(round(op_lat_deg * 1e7)))
    struct.pack_into('<i', msg, 6, int(round(op_lon_deg * 1e7)))
    struct.pack_into('<H', msg, 10, 0)          # AreaCount
    msg[12] = 0                                  # AreaRadius
    struct.pack_into('<H', msg, 13, 0)          # AreaCeiling
    struct.pack_into('<H', msg, 15, 0)          # AreaFloor
    msg[17] = category_class_byte & 0xFF        # Category/Class
    struct.pack_into('<H', msg, 18, _encode_alt(operator_alt_m))
    struct.pack_into('<I', msg, 20, 0)          # Timestamp
    msg[24] = 0                                  # Reserved
    return bytes(msg)


class TestODIDSystemParser(unittest.TestCase):
    def test_operator_altitude_decodes_at_correct_offset(self):
        device = DroneIDDevice()
        ODIDParser.parse_system(_build_system_msg(operator_alt_m=24.0), device)
        self.assertAlmostEqual(device.operator_alt, 24.0, places=1)

    def test_category_byte_does_not_leak_into_altitude(self):
        # A large category/class byte must NOT perturb the decoded altitude.
        device = DroneIDDevice()
        ODIDParser.parse_system(
            _build_system_msg(operator_alt_m=100.0, category_class_byte=0xFF),
            device,
        )
        self.assertAlmostEqual(device.operator_alt, 100.0, places=1)

    def test_ground_level_operator_is_not_kilometers(self):
        # Regression: the off-by-one produced ~24 km for a ground operator.
        device = DroneIDDevice()
        ODIDParser.parse_system(_build_system_msg(operator_alt_m=0.0), device)
        self.assertLess(abs(device.operator_alt), 50.0)

    def test_operator_latlon_decode(self):
        device = DroneIDDevice()
        ODIDParser.parse_system(
            _build_system_msg(op_lat_deg=33.14097, op_lon_deg=-80.10749),
            device,
        )
        self.assertAlmostEqual(device.operator_lat, 33.14097, places=4)
        self.assertAlmostEqual(device.operator_lon, -80.10749, places=4)

    def test_altitude_sentinels_ignored(self):
        # 0x0000 and 0xFFFF at bytes 18-19 mean "unknown" -> leave default.
        for raw in (0x0000, 0xFFFF):
            device = DroneIDDevice()
            msg = bytearray(_build_system_msg())
            struct.pack_into('<H', msg, 18, raw)
            ODIDParser.parse_system(bytes(msg), device)
            self.assertEqual(device.operator_alt, 0.0)


if __name__ == '__main__':
    unittest.main()
