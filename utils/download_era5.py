"""Download ERA5 pressure/single pairs for CAESAR and other ERA5 tests."""
import argparse
from contextlib import contextmanager
from datetime import date, datetime, timedelta
import fcntl
import os
import re
import time as time_module
import zipfile
from collections import Counter
from pathlib import Path

import cdsapi
import requests

# CDS API settings (from cra5/api/era5_downloader.py)
os.environ["CDSAPI_URL"] = "https://cds.climate.copernicus.eu/api"
os.environ["CDSAPI_KEY"] = "ea3a2607-158c-48a4-bd27-b255256b2759"

DEFAULT_DATA_DIR = Path("/data/run01/scxj523/zsh/project/Data/ERA5/2024")

PRESSURE_VARIABLES = [
    "geopotential",
    "specific_humidity",
    "u_component_of_wind",
    "v_component_of_wind",
    "temperature",
    "relative_humidity",
    "vertical_velocity",
]

PRESSURE_LEVELS = [
    "1", "2", "3", "5", "7", "10", "20", "30", "50", "70",
    "100", "125", "150", "175", "200", "225", "250",
    "300", "350", "400", "450", "500", "550", "600", "650", "700",
    "750", "775", "800", "825", "850", "875", "900", "925", "950", "975", "1000",
]

SINGLE_VARIABLES = [
    "2m_temperature",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "100m_u_component_of_wind",
    "100m_v_component_of_wind",
    "total_cloud_cover",
    "surface_pressure",
    "total_precipitation",
    "mean_sea_level_pressure",
]


@contextmanager
def file_lock(lock_path):
    """Take an advisory inter-process lock at a stable path."""
    lock_path = Path(lock_path)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("w") as lock_stream:
        fcntl.flock(lock_stream, fcntl.LOCK_EX)
        yield


@contextmanager
def output_lock(outfile):
    """Serialize workers that may reach the same daily output."""
    with file_lock(Path(f"{outfile}.lock")):
        yield


def build_days(start_date, count):
    start = date.fromisoformat(start_date)
    return [start + timedelta(days=idx) for idx in range(count)]


def group_days_for_pressure_requests(days, max_days):
    """Group contiguous dates without crossing a calendar month."""
    groups = []
    current = []
    for day in days:
        if current and (
            len(current) >= max_days
            or day != current[-1] + timedelta(days=1)
            or (day.year, day.month) != (current[0].year, current[0].month)
        ):
            groups.append(current)
            current = []
        current.append(day)
    if current:
        groups.append(current)
    return groups


def resolve_times(time, times):
    if times is None:
        resolved = [time]
    elif len(times) == 1 and times[0].lower() == "hourly":
        resolved = [f"{hour:02d}:00" for hour in range(24)]
    else:
        resolved = times

    for value in resolved:
        datetime.strptime(value, "%H:%M")
    return list(dict.fromkeys(resolved))


def output_paths(data_dir, day, time):
    timestamp = f"{day.isoformat()}T{time}:00"
    data_dir = Path(data_dir)
    return data_dir / f"{timestamp}_pressure.nc", data_dir / f"{timestamp}_single.nc"


def batch_output_paths(data_dir, day, times):
    if times == [f"{hour:02d}:00" for hour in range(24)]:
        label = "hourly"
    else:
        label = f"{times[0].replace(':', '')}-{times[-1].replace(':', '')}-{len(times)}t"
    data_dir = Path(data_dir)
    stem = f"{day.isoformat()}_{label}"
    return data_dir / f"{stem}_pressure.nc", data_dir / f"{stem}_single.nc"


def pressure_output_path(path, data_format):
    path = Path(path)
    if data_format == "netcdf":
        return path
    if data_format == "grib":
        return path.with_suffix(".grib")
    raise ValueError(f"Unsupported pressure data format: {data_format}")


def alternate_pressure_path(path):
    path = Path(path)
    if path.suffix == ".grib":
        return path.with_suffix(".nc")
    if path.suffix == ".nc":
        return path.with_suffix(".grib")
    raise ValueError(f"Unsupported pressure output suffix: {path}")


def pressure_day_exists(outfile):
    outfile = Path(outfile)
    if outfile.exists() and outfile.stat().st_size > 1_000_000:
        return True
    alternate = alternate_pressure_path(outfile)
    return alternate.exists() and alternate.stat().st_size > 1_000_000


def split_grib_by_day(source, outputs, expected_messages_per_day):
    """Split a multi-day GRIB stream into validated, atomically published days."""
    import eccodes

    source = Path(source)
    outputs = {day: Path(path) for day, path in outputs.items()}
    temporary = {
        day: path.with_suffix(path.suffix + ".split.tmp")
        for day, path in outputs.items()
    }
    streams = {}
    counts = Counter()
    expected_keys = {int(day.strftime("%Y%m%d")): day for day in outputs}

    for path in temporary.values():
        path.unlink(missing_ok=True)

    try:
        streams = {
            day: temporary[day].open("wb")
            for day in sorted(temporary)
        }
        with source.open("rb") as input_stream:
            while True:
                handle = eccodes.codes_grib_new_from_file(input_stream)
                if handle is None:
                    break
                try:
                    data_date = int(eccodes.codes_get(handle, "dataDate"))
                    if data_date not in expected_keys:
                        raise ValueError(
                            f"Unexpected dataDate={data_date} in {source}; "
                            f"expected {sorted(expected_keys)}"
                        )
                    day = expected_keys[data_date]
                    streams[day].write(eccodes.codes_get_message(handle))
                    counts[day] += 1
                finally:
                    eccodes.codes_release(handle)
    finally:
        for stream in streams.values():
            stream.close()

    wrong = {
        day.isoformat(): counts[day]
        for day in outputs
        if counts[day] != expected_messages_per_day
    }
    empty = [
        day.isoformat()
        for day, path in temporary.items()
        if not path.exists() or path.stat().st_size <= 1_000_000
    ]
    if wrong or empty:
        for path in temporary.values():
            path.unlink(missing_ok=True)
        raise ValueError(
            f"Invalid multi-day GRIB split for {source}: "
            f"message_counts={wrong}, empty={empty}, "
            f"expected_per_day={expected_messages_per_day}"
        )

    for day in sorted(outputs):
        temporary[day].replace(outputs[day])
    return {day: counts[day] for day in outputs}


def extract_or_move(tmp_file, outfile, data_dir):
    tmp_file = Path(tmp_file)
    outfile = Path(outfile)
    data_dir = Path(data_dir)

    if zipfile.is_zipfile(tmp_file):
        print("Downloaded file is a zip, extracting...")
        import xarray as xr

        with zipfile.ZipFile(tmp_file, "r") as zf:
            nc_files = [name for name in zf.namelist() if name.endswith(".nc")]
            print(f"  Found {len(nc_files)} nc files in zip: {nc_files}")
            zf.extractall(data_dir, members=nc_files)
            extracted_paths = [data_dir / name for name in nc_files]

        if len(extracted_paths) == 1:
            extracted_paths[0].replace(outfile)
        else:
            datasets = [xr.open_dataset(path) for path in extracted_paths]
            try:
                merged = xr.merge(datasets)
                try:
                    # Keep source datasets open until lazy variables are written.
                    merged.to_netcdf(outfile)
                finally:
                    merged.close()
                print(f"  Merged {len(extracted_paths)} files into {outfile}")
            finally:
                for ds in datasets:
                    ds.close()
                for path in extracted_paths:
                    if path.exists() and path != outfile:
                        path.unlink()
        tmp_file.unlink()
    else:
        tmp_file.replace(outfile)


def download_result_resumable(
    result,
    tmp_file,
    max_retries,
    retry_wait,
    session=None,
    chunk_size=8 * 1024 * 1024,
):
    """Download one completed CDS result without discarding partial bytes."""
    tmp_file = Path(tmp_file)
    expected_size = int(result.content_length)
    url = result.location
    session = session or requests.Session()

    if tmp_file.exists() and tmp_file.stat().st_size > expected_size:
        raise RuntimeError(
            f"Partial file is larger than the CDS result: "
            f"{tmp_file.stat().st_size} > {expected_size}"
        )

    for attempt in range(max_retries + 1):
        offset = tmp_file.stat().st_size if tmp_file.exists() else 0
        if offset == expected_size:
            return

        headers = {"Range": f"bytes={offset}-"} if offset else {}
        response = None
        try:
            response = session.get(
                url,
                headers=headers,
                stream=True,
                allow_redirects=True,
                timeout=(30, 300),
            )
            response.raise_for_status()

            if offset:
                content_range = response.headers.get("Content-Range", "")
                match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", content_range)
                if response.status_code != 206 or not match:
                    raise RuntimeError(
                        "CDS object store ignored the byte-range request; "
                        f"status={response.status_code}, Content-Range={content_range!r}"
                    )
                start, _, total = map(int, match.groups())
                if start != offset or total != expected_size:
                    raise RuntimeError(
                        "Unexpected CDS byte range: "
                        f"start={start}, total={total}, expected "
                        f"start={offset}, total={expected_size}"
                    )

            mode = "ab" if offset else "wb"
            with tmp_file.open(mode) as stream:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        stream.write(chunk)

            actual_size = tmp_file.stat().st_size
            if actual_size == expected_size:
                return
            raise IOError(
                f"Incomplete CDS download: {actual_size}/{expected_size} bytes"
            )
        except Exception as exc:
            if attempt >= max_retries:
                raise
            delay = min(retry_wait * (2 ** min(attempt, 4)), 120)
            saved = tmp_file.stat().st_size if tmp_file.exists() else 0
            print(
                f"CDS transfer failed ({type(exc).__name__}: {exc}); "
                f"preserved {saved}/{expected_size} bytes, retrying in {delay}s "
                f"[{attempt + 1}/{max_retries}]",
                flush=True,
            )
            time_module.sleep(delay)
        finally:
            if response is not None:
                response.close()


def retrieve_with_retry(client, dataset, request, tmp_file, max_retries, retry_wait):
    submit_lock = Path(tmp_file).parent / f".cds_{dataset}.submit.lock"
    for attempt in range(max_retries + 1):
        try:
            # CDS limits queued requests per dataset. Hold this lock only until
            # the result is ready, then overlap its object-store transfer with
            # generation of the next day.
            with file_lock(submit_lock):
                result = client.retrieve(dataset, request)
            download_result_resumable(
                result,
                tmp_file,
                max_retries=max_retries,
                retry_wait=min(retry_wait, 30),
            )
            return
        except Exception as exc:
            if attempt >= max_retries:
                raise
            delay = min(retry_wait * (2 ** min(attempt, 4)), 3600)
            print(
                f"CDS request failed ({type(exc).__name__}: {exc}); "
                f"retrying in {delay}s [{attempt + 1}/{max_retries}]",
                flush=True,
            )
            time_module.sleep(delay)


def download_pressure(
    client,
    data_dir,
    day,
    times,
    outfile=None,
    area=None,
    max_retries=20,
    retry_wait=300,
    data_format="netcdf",
):
    if isinstance(times, str):
        times = [times]
    if outfile is None:
        outfile, _ = output_paths(data_dir, day, times[0])
        outfile = pressure_output_path(outfile, data_format)
    outfile = Path(outfile)
    if outfile.exists() and outfile.stat().st_size > 1_000_000:
        print(f"Pressure file already exists: {outfile} ({outfile.stat().st_size} bytes)")
        return
    alternate = alternate_pressure_path(outfile)
    if alternate.exists() and alternate.stat().st_size > 1_000_000:
        print(
            f"Equivalent pressure file already exists: {alternate} "
            f"({alternate.stat().st_size} bytes)"
        )
        return

    print(f"Downloading pressure level data for {day.isoformat()} {times}...")
    print(f"  Variables: {PRESSURE_VARIABLES}")
    print(f"  Levels: {len(PRESSURE_LEVELS)} levels")
    print(f"  Output: {outfile}")

    request = {
        "product_type": ["reanalysis"],
        "variable": PRESSURE_VARIABLES,
        "pressure_level": PRESSURE_LEVELS,
        "year": day.strftime("%Y"),
        "month": day.strftime("%m"),
        "day": day.strftime("%d"),
        "time": times,
        "data_format": data_format,
    }
    if area is not None:
        request["area"] = area

    tmp_file = outfile.with_suffix(outfile.suffix + ".tmp")
    retrieve_with_retry(
        client,
        "reanalysis-era5-pressure-levels",
        request,
        tmp_file,
        max_retries,
        retry_wait,
    )
    extract_or_move(tmp_file, outfile, data_dir)
    print(f"Done! File size: {outfile.stat().st_size} bytes")


def download_pressure_days(
    client,
    data_dir,
    days,
    times,
    area=None,
    max_retries=20,
    retry_wait=300,
):
    """Retrieve several same-month days as one GRIB and publish daily files."""
    days = list(days)
    if not days:
        return
    if len({(day.year, day.month) for day in days}) != 1:
        raise ValueError("A multi-day ERA5 request cannot cross a calendar month")

    outputs = {}
    for day in days:
        pressure_out, _ = batch_output_paths(data_dir, day, times)
        output = pressure_output_path(pressure_out, "grib")
        if pressure_day_exists(output):
            print(f"Equivalent pressure day already exists: {day.isoformat()}")
        else:
            outputs[day] = output
    if not outputs:
        return

    request_days = sorted(outputs)
    first, last = request_days[0], request_days[-1]
    batch_file = Path(data_dir) / (
        f".pressure_batch_{first.isoformat()}_{last.isoformat()}_"
        f"{len(request_days)}d.grib"
    )
    expected_per_day = len(PRESSURE_VARIABLES) * len(PRESSURE_LEVELS) * len(times)
    request_items = expected_per_day * len(request_days)
    if request_items > 60_000:
        raise ValueError(
            f"Pressure request has {request_items} items, above the 60,000-item "
            "safety limit"
        )

    print(
        f"Downloading {len(request_days)} pressure days in one native GRIB "
        f"({first} through {last}, {request_items} items)...",
        flush=True,
    )
    request = {
        "product_type": ["reanalysis"],
        "variable": PRESSURE_VARIABLES,
        "pressure_level": PRESSURE_LEVELS,
        "year": first.strftime("%Y"),
        "month": first.strftime("%m"),
        "day": [day.strftime("%d") for day in request_days],
        "time": times,
        "data_format": "grib",
    }
    if area is not None:
        request["area"] = area

    with output_lock(batch_file):
        if not batch_file.exists() or batch_file.stat().st_size <= 1_000_000:
            tmp_file = batch_file.with_suffix(batch_file.suffix + ".tmp")
            retrieve_with_retry(
                client,
                "reanalysis-era5-pressure-levels",
                request,
                tmp_file,
                max_retries,
                retry_wait,
            )
            tmp_file.replace(batch_file)
        counts = split_grib_by_day(
            batch_file,
            outputs,
            expected_messages_per_day=expected_per_day,
        )
        batch_file.unlink()

    for day in request_days:
        print(
            f"Published {outputs[day]} "
            f"({outputs[day].stat().st_size} bytes, {counts[day]} messages)",
            flush=True,
        )


def download_single(
    client,
    data_dir,
    day,
    times,
    outfile=None,
    area=None,
    max_retries=20,
    retry_wait=300,
):
    if isinstance(times, str):
        times = [times]
    if outfile is None:
        _, outfile = output_paths(data_dir, day, times[0])
    if outfile.exists() and outfile.stat().st_size > 100_000:
        print(f"Single file already exists: {outfile} ({outfile.stat().st_size} bytes)")
        return

    print(f"Downloading single level data for {day.isoformat()} {times}...")
    print(f"  Variables: {SINGLE_VARIABLES}")
    print(f"  Output: {outfile}")

    request = {
        "product_type": ["reanalysis"],
        "variable": SINGLE_VARIABLES,
        "year": day.strftime("%Y"),
        "month": day.strftime("%m"),
        "day": day.strftime("%d"),
        "time": times,
        "data_format": "netcdf",
    }
    if area is not None:
        request["area"] = area

    tmp_file = outfile.with_suffix(outfile.suffix + ".tmp")
    retrieve_with_retry(
        client,
        "reanalysis-era5-single-levels",
        request,
        tmp_file,
        max_retries,
        retry_wait,
    )
    extract_or_move(tmp_file, outfile, data_dir)
    print(f"Done! File size: {outfile.stat().st_size} bytes")


def verify_data(data_dir, day, time):
    """Verify the downloaded data has correct structure."""
    try:
        import xarray as xr
        pressure_f, single_f = output_paths(data_dir, day, time)

        if not pressure_f.exists():
            print(f"MISSING: {pressure_f}")
            return False
        if not single_f.exists():
            print(f"MISSING: {single_f}")
            return False

        ds_p = xr.open_dataset(pressure_f)
        ds_s = xr.open_dataset(single_f)

        print("\n=== Pressure file ===")
        print(f"  Variables: {list(ds_p.data_vars)}")
        print(f"  Pressure levels: {len(ds_p.pressure_level) if 'pressure_level' in ds_p.sizes else 'N/A'}")
        print(f"  Shape: lat={len(ds_p.latitude)}, lon={len(ds_p.longitude)}")

        print("\n=== Single file ===")
        print(f"  Variables: {list(ds_s.data_vars)}")
        print(f"  Shape: lat={len(ds_s.latitude)}, lon={len(ds_s.longitude)}")

        n_pressure = len(list(ds_p.data_vars)) * len(ds_p.pressure_level)
        n_single = len(list(ds_s.data_vars))
        total = n_pressure + n_single
        print(f"\nTotal channels: {n_pressure} (pressure) + {n_single} (single) = {total}")

        ds_p.close()
        ds_s.close()
        return True
    except Exception as e:
        print(f"Verification error: {e}")
        return False


def parse_args():
    parser = argparse.ArgumentParser(description="Download contiguous ERA5 pressure/single nc pairs.")
    parser.add_argument("--start-date", default="2024-06-01", help="Inclusive start date, YYYY-MM-DD.")
    parser.add_argument("--days", type=int, default=16, help="Number of contiguous days to download.")
    parser.add_argument(
        "--time",
        default="00:00",
        help="One ERA5 time per day, HH:MM. Ignored when --times is supplied.",
    )
    parser.add_argument(
        "--times",
        nargs="+",
        help='ERA5 times per day as HH:MM values, or the single value "hourly".',
    )
    parser.add_argument(
        "--batch-times",
        action="store_true",
        help="Request all selected times for a day together and write one daily file pair.",
    )
    parser.add_argument(
        "--area",
        nargs=4,
        type=float,
        metavar=("NORTH", "WEST", "SOUTH", "EAST"),
        help="Optional CDS subset in north west south east order.",
    )
    parser.add_argument("--max-retries", type=int, default=20)
    parser.add_argument("--retry-wait", type=int, default=300)
    parser.add_argument(
        "--content",
        choices=["both", "pressure", "single"],
        default="both",
        help="Download both collections or only one collection.",
    )
    parser.add_argument(
        "--pressure-format",
        choices=["netcdf", "grib"],
        default="netcdf",
        help="Request pressure-level data as converted NetCDF or native GRIB.",
    )
    parser.add_argument(
        "--pressure-days-per-request",
        type=int,
        default=1,
        help=(
            "Combine this many same-month pressure days in each native-GRIB "
            "request, then split and validate daily files (1-9)."
        ),
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--skip-verify", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    times = resolve_times(args.time, args.times)
    if not 1 <= args.pressure_days_per_request <= 9:
        raise ValueError("--pressure-days-per-request must be between 1 and 9")
    if args.pressure_days_per_request > 1 and (
        args.pressure_format != "grib"
        or not args.batch_times
        or args.content != "pressure"
    ):
        raise ValueError(
            "Multi-day pressure requests require --pressure-format grib, "
            "--batch-times, and --content pressure"
        )
    args.data_dir.mkdir(parents=True, exist_ok=True)
    client = cdsapi.Client()

    days = build_days(args.start_date, args.days)
    if args.pressure_days_per_request > 1:
        groups = group_days_for_pressure_requests(
            days, args.pressure_days_per_request
        )
        for group in groups:
            print(f"\n{'=' * 60}")
            print(
                f"Downloading pressure batch "
                f"{group[0].isoformat()} through {group[-1].isoformat()}"
            )
            print(f"{'=' * 60}")
            download_pressure_days(
                client,
                args.data_dir,
                group,
                times,
                args.area,
                args.max_retries,
                args.retry_wait,
            )
        return

    for day in days:
        if args.batch_times:
            pressure_out, single_out = batch_output_paths(args.data_dir, day, times)
            pressure_out = pressure_output_path(
                pressure_out, args.pressure_format
            )
            print(f"\n{'=' * 60}")
            print(f"Downloading {len(times)} batched times for {day.isoformat()}")
            print(f"{'=' * 60}")
            if args.content in ("both", "pressure"):
                with output_lock(pressure_out):
                    download_pressure(
                        client,
                        args.data_dir,
                        day,
                        times,
                        pressure_out,
                        args.area,
                        args.max_retries,
                        args.retry_wait,
                        args.pressure_format,
                    )
            if args.content in ("both", "single"):
                with output_lock(single_out):
                    download_single(
                        client,
                        args.data_dir,
                        day,
                        times,
                        single_out,
                        args.area,
                        args.max_retries,
                        args.retry_wait,
                    )
            continue

        for time in times:
            print(f"\n{'=' * 60}")
            print(f"Downloading data for {day.isoformat()} {time}")
            print(f"{'=' * 60}")
            if args.content in ("both", "pressure"):
                pressure_out, _ = output_paths(args.data_dir, day, time)
                pressure_out = pressure_output_path(
                    pressure_out, args.pressure_format
                )
                with output_lock(pressure_out):
                    download_pressure(
                        client, args.data_dir, day, time, area=args.area,
                        max_retries=args.max_retries, retry_wait=args.retry_wait,
                        data_format=args.pressure_format,
                    )
            if args.content in ("both", "single"):
                _, single_out = output_paths(args.data_dir, day, time)
                with output_lock(single_out):
                    download_single(
                        client, args.data_dir, day, time, area=args.area,
                        max_retries=args.max_retries, retry_wait=args.retry_wait,
                    )
            if not args.skip_verify:
                verify_data(args.data_dir, day, time)


if __name__ == "__main__":
    main()
