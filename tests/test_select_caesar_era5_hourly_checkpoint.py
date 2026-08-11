import unittest

from scripts.select_caesar_era5_hourly_checkpoint import piecewise_log_bd_rate


def _curve(rate_scale=1.0):
    return [
        {"scientific_bpp": rate_scale * rate, "psnr": psnr}
        for rate, psnr in ((0.2, 40.0), (0.5, 50.0), (1.0, 60.0))
    ]


class SelectCaesarEra5HourlyCheckpointTest(unittest.TestCase):
    def test_identical_curve_has_zero_bd_rate(self):
        self.assertAlmostEqual(piecewise_log_bd_rate(_curve(), _curve()), 0.0)

    def test_uniformly_lower_rate_curve_is_preferred(self):
        self.assertAlmostEqual(
            piecewise_log_bd_rate(_curve(), _curve(0.9)),
            -10.0,
            places=6,
        )


if __name__ == "__main__":
    unittest.main()
