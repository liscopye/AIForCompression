import importlib.util
import sys
import tempfile
import types
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "utils" / "download_era5.py"


def load_download_module():
    sys.modules.setdefault("cdsapi", types.SimpleNamespace(Client=object))
    spec = importlib.util.spec_from_file_location("download_era5", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class DownloadEra5Test(unittest.TestCase):
    def test_build_days_from_start_date_returns_contiguous_dates(self):
        module = load_download_module()

        days = module.build_days("2024-06-01", 8)

        self.assertEqual(
            ["2024-06-01", "2024-06-02", "2024-06-03", "2024-06-04"],
            [day.isoformat() for day in days[:4]],
        )
        self.assertEqual("2024-06-08", days[-1].isoformat())

    def test_group_days_for_pressure_requests_does_not_cross_month(self):
        module = load_download_module()
        days = module.build_days("2024-03-28", 10)

        groups = module.group_days_for_pressure_requests(days, 4)

        self.assertEqual(
            [
                ["2024-03-28", "2024-03-29", "2024-03-30", "2024-03-31"],
                ["2024-04-01", "2024-04-02", "2024-04-03", "2024-04-04"],
                ["2024-04-05", "2024-04-06"],
            ],
            [[day.isoformat() for day in group] for group in groups],
        )

    def test_output_paths_match_caesar_pair_naming(self):
        module = load_download_module()
        day = module.build_days("2024-06-01", 1)[0]

        pressure, single = module.output_paths(Path("/tmp/era5"), day, "00:00")

        self.assertEqual(Path("/tmp/era5/2024-06-01T00:00:00_pressure.nc"), pressure)
        self.assertEqual(Path("/tmp/era5/2024-06-01T00:00:00_single.nc"), single)

    def test_hourly_batch_paths_and_request(self):
        module = load_download_module()
        day = module.build_days("2024-03-01", 1)[0]
        times = module.resolve_times("00:00", ["hourly"])
        pressure, single = module.batch_output_paths(Path("/tmp/era5"), day, times)

        self.assertEqual(24, len(times))
        self.assertEqual(Path("/tmp/era5/2024-03-01_hourly_pressure.nc"), pressure)
        self.assertEqual(Path("/tmp/era5/2024-03-01_hourly_single.nc"), single)

        class FakeClient:
            def __init__(self):
                self.calls = []

            def retrieve(self, dataset, request):
                self.calls.append((dataset, request))
                return types.SimpleNamespace(
                    content_length=6,
                    location="https://example.invalid/data.nc",
                )

        class FakeResponse:
            status_code = 200
            headers = {}

            def raise_for_status(self):
                return None

            def iter_content(self, chunk_size):
                yield b"netcdf"

            def close(self):
                return None

        class FakeSession:
            def get(self, *args, **kwargs):
                return FakeResponse()

        client = FakeClient()
        with self.subTest("pressure request"):
            with tempfile.TemporaryDirectory() as directory:
                output = Path(directory) / pressure.name
                original = module.requests.Session
                module.requests.Session = FakeSession
                try:
                    module.download_pressure(
                        client,
                        Path(directory),
                        day,
                        times,
                        outfile=output,
                        area=[64, -64, -64, 64],
                        max_retries=0,
                    )
                finally:
                    module.requests.Session = original
                self.assertTrue(output.is_file())

        dataset, request = client.calls[-1]
        self.assertEqual("reanalysis-era5-pressure-levels", dataset)
        self.assertEqual(times, request["time"])
        self.assertEqual([64, -64, -64, 64], request["area"])
        self.assertEqual("netcdf", request["data_format"])

    def test_grib_pressure_path_and_existing_netcdf_skip(self):
        module = load_download_module()
        day = module.build_days("2024-03-01", 1)[0]
        times = module.resolve_times("00:00", ["hourly"])
        pressure, _ = module.batch_output_paths(Path("/tmp/era5"), day, times)
        grib = module.pressure_output_path(pressure, "grib")

        self.assertEqual(
            Path("/tmp/era5/2024-03-01_hourly_pressure.grib"), grib
        )
        self.assertEqual(pressure, module.alternate_pressure_path(grib))

        class UnexpectedClient:
            def retrieve(self, *args, **kwargs):
                raise AssertionError("existing NetCDF date must not be re-requested")

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            existing = root / pressure.name
            existing.write_bytes(b"\0" * 1_000_001)
            module.download_pressure(
                UnexpectedClient(),
                root,
                day,
                times,
                outfile=root / grib.name,
                data_format="grib",
            )

    def test_download_pressure_days_requests_missing_dates_and_splits(self):
        module = load_download_module()
        days = module.build_days("2024-03-01", 3)
        times = module.resolve_times("00:00", ["hourly"])
        requests = []
        expected_counts = []

        class FakeClient:
            def retrieve(self, dataset, request):
                requests.append((dataset, request))
                return types.SimpleNamespace(
                    content_length=9,
                    location="https://example.invalid/data.grib",
                )

        class FakeResponse:
            status_code = 200
            headers = {}

            def raise_for_status(self):
                return None

            def iter_content(self, chunk_size):
                yield b"fake-grib"

            def close(self):
                return None

        class FakeSession:
            def get(self, *args, **kwargs):
                return FakeResponse()

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            existing = root / "2024-03-02_hourly_pressure.nc"
            existing.write_bytes(b"\0" * 1_000_001)
            original_session = module.requests.Session
            original_split = module.split_grib_by_day

            def fake_split(source, outputs, expected_messages_per_day):
                self.assertEqual({days[0], days[2]}, set(outputs))
                expected_counts.append(expected_messages_per_day)
                for output in outputs.values():
                    output.write_bytes(b"\0" * 1_000_001)
                return {
                    day: expected_messages_per_day
                    for day in outputs
                }

            module.requests.Session = FakeSession
            module.split_grib_by_day = fake_split
            try:
                module.download_pressure_days(
                    FakeClient(),
                    root,
                    days,
                    times,
                    area=[64, -64, -64, 64],
                    max_retries=0,
                )
            finally:
                module.requests.Session = original_session
                module.split_grib_by_day = original_split

            self.assertFalse(
                any(root.glob(".pressure_batch_*.grib"))
            )
            self.assertTrue(
                (root / "2024-03-01_hourly_pressure.grib").is_file()
            )
            self.assertTrue(
                (root / "2024-03-03_hourly_pressure.grib").is_file()
            )

        dataset, request = requests[0]
        self.assertEqual("reanalysis-era5-pressure-levels", dataset)
        self.assertEqual(["01", "03"], request["day"])
        self.assertEqual("grib", request["data_format"])
        self.assertEqual([7 * 37 * 24], expected_counts)

    def test_resumable_download_appends_valid_byte_range(self):
        module = load_download_module()

        class FakeResponse:
            status_code = 206
            headers = {"Content-Range": "bytes 4-9/10"}

            def raise_for_status(self):
                return None

            def iter_content(self, chunk_size):
                yield b"456"
                yield b"789"

            def close(self):
                return None

        class FakeSession:
            def __init__(self):
                self.headers = None

            def get(self, url, **kwargs):
                self.headers = kwargs["headers"]
                return FakeResponse()

        with tempfile.TemporaryDirectory() as directory:
            partial = Path(directory) / "result.tmp"
            partial.write_bytes(b"0123")
            session = FakeSession()
            result = types.SimpleNamespace(
                content_length=10,
                location="https://example.invalid/result.nc",
            )

            module.download_result_resumable(
                result,
                partial,
                max_retries=0,
                retry_wait=0,
                session=session,
            )

            self.assertEqual({"Range": "bytes=4-"}, session.headers)
            self.assertEqual(b"0123456789", partial.read_bytes())

    def test_resumable_download_does_not_overwrite_when_range_is_ignored(self):
        module = load_download_module()

        class FakeResponse:
            status_code = 200
            headers = {}

            def raise_for_status(self):
                return None

            def close(self):
                return None

        class FakeSession:
            def get(self, *args, **kwargs):
                return FakeResponse()

        with tempfile.TemporaryDirectory() as directory:
            partial = Path(directory) / "result.tmp"
            partial.write_bytes(b"keep")
            result = types.SimpleNamespace(
                content_length=10,
                location="https://example.invalid/result.nc",
            )

            with self.assertRaisesRegex(RuntimeError, "ignored the byte-range"):
                module.download_result_resumable(
                    result,
                    partial,
                    max_retries=0,
                    retry_wait=0,
                    session=FakeSession(),
                )

            self.assertEqual(b"keep", partial.read_bytes())

    def test_resumable_download_preserves_new_bytes_across_interruption(self):
        module = load_download_module()

        class InterruptedResponse:
            status_code = 200
            headers = {}

            def raise_for_status(self):
                return None

            def iter_content(self, chunk_size):
                yield b"0123"
                raise OSError("connection interrupted")

            def close(self):
                return None

        class ResumedResponse:
            status_code = 206
            headers = {"Content-Range": "bytes 4-9/10"}

            def raise_for_status(self):
                return None

            def iter_content(self, chunk_size):
                yield b"456789"

            def close(self):
                return None

        class FakeSession:
            def __init__(self):
                self.calls = []

            def get(self, url, **kwargs):
                self.calls.append(kwargs["headers"])
                if len(self.calls) == 1:
                    return InterruptedResponse()
                return ResumedResponse()

        with tempfile.TemporaryDirectory() as directory:
            partial = Path(directory) / "result.tmp"
            result = types.SimpleNamespace(
                content_length=10,
                location="https://example.invalid/result.nc",
            )
            session = FakeSession()

            module.download_result_resumable(
                result,
                partial,
                max_retries=1,
                retry_wait=0,
                session=session,
            )

            self.assertEqual([{}, {"Range": "bytes=4-"}], session.calls)
            self.assertEqual(b"0123456789", partial.read_bytes())

    def test_extract_zip_keeps_sources_open_while_merging(self):
        module = load_download_module()
        import tempfile
        import zipfile

        import netCDF4

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sources = []
            for name, variable, value in (
                ("instant.nc", "t2m", 1.0),
                ("accum.nc", "tp", 2.0),
            ):
                path = root / name
                with netCDF4.Dataset(path, "w") as ds:
                    ds.createDimension("valid_time", 1)
                    ds.createVariable(variable, "f4", ("valid_time",))[:] = [value]
                sources.append(path)

            archive = root / "download.tmp"
            with zipfile.ZipFile(archive, "w") as zf:
                for source in sources:
                    zf.write(source, source.name)
                    source.unlink()

            output = root / "merged.nc"
            module.extract_or_move(archive, output, root)
            with netCDF4.Dataset(output, "r") as ds:
                self.assertEqual({"t2m", "tp"}, set(ds.variables))
                self.assertEqual(1.0, float(ds.variables["t2m"][0]))
                self.assertEqual(2.0, float(ds.variables["tp"][0]))


if __name__ == "__main__":
    unittest.main()
