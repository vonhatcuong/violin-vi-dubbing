import unittest

from pipeline import tts_edge


class EdgeTtsBackendTests(unittest.TestCase):
    def test_vietnamese_native_voices_are_available(self):
        self.assertEqual(
            ["vi-VN-NamMinhNeural", "vi-VN-HoaiMyNeural"],
            tts_edge.native_voices_for("vi"),
        )

    def test_speed_maps_to_edge_rate_percent(self):
        self.assertEqual("+10%", tts_edge._speed_to_rate(1.1))
        self.assertEqual("-8%", tts_edge._speed_to_rate(0.92))
        self.assertEqual("+0%", tts_edge._speed_to_rate(None))


if __name__ == "__main__":
    unittest.main()
