from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from pystac_client import Client
from rasterio.enums import Resampling
from rasterio.windows import from_bounds
from rasterio.warp import transform_bounds


CITY_BBOXES = {
    "boston": [-71.1912, 42.2279, -70.9860, 42.3995],
    "new_york": [-74.2591, 40.4774, -73.7004, 40.9176],
    "buffalo": [-78.9370, 42.8261, -78.7382, 42.9585],
    "chicago": [-87.9401, 41.6445, -87.5237, 42.0230],
    "denver": [-105.1099, 39.6144, -104.6003, 39.9142],
    "minneapolis": [-93.3800, 44.8650, -93.1550, 45.0600],
}

EARTH_SEARCH_URL = "https://earth-search.aws.element84.com/v1"
DEFAULT_BANDS = ["B02", "B03", "B04", "B08", "B11", "B12"]
IGNORE_LABEL = 255

CLOUD_CLASSES = {3, 8, 9, 10}
WATER_CLASS = 6
SNOW_CLASS = 11
NODATA_CLASSES = {0, 1}

ASSET_KEYS = {
    "B02": ["B02", "blue"],
    "B03": ["B03", "green"],
    "B04": ["B04", "red"],
    "B08": ["B08", "nir"],
    "B11": ["B11", "swir16"],
    "B12": ["B12", "swir22"],
    "SCL": ["SCL", "scl"],
    "CLD": ["CLD", "cld", "MSK_CLDPRB"],
    "SNW": ["SNW", "snw", "MSK_SNWPRB"],
}


@dataclass
class SplitConfig:
    train: float
    val: float
    test: float
    unit: str


@dataclass
class PatchConfig:
    size: int
    stride: int
    min_labeled_ratio: float
    min_clear_ratio: float
    background_keep_prob: float
    min_snow_ratio_positive: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a CNN-ready Sentinel-2 snow dataset.")
    parser.add_argument("--cities", nargs="+", default=["boston"], help="City names or 'all'.")
    parser.add_argument("--city-file", default=None, help="Optional JSON file with custom city bboxes.")
    parser.add_argument("--years", nargs="*", type=int, help="Explicit list of anchor years.")
    parser.add_argument("--year-start", type=int, default=None)
    parser.add_argument("--year-end", type=int, default=None)
    parser.add_argument("--start-month-day", default="12-01")
    parser.add_argument("--end-month-day", default="02-28")
    parser.add_argument("--max-items-per-window", type=int, default=12)
    parser.add_argument("--max-cloud", type=float, default=30.0)
    parser.add_argument("--bands", nargs="+", default=DEFAULT_BANDS)
    parser.add_argument("--patch-size", type=int, default=256)
    parser.add_argument("--stride", type=int, default=256)
    parser.add_argument("--min-labeled-ratio", type=float, default=0.50)
    parser.add_argument("--min-clear-ratio", type=float, default=0.50)
    parser.add_argument("--background-keep-prob", type=float, default=0.20)
    parser.add_argument("--min-snow-ratio-positive", type=float, default=0.01)
    parser.add_argument("--ndsi-threshold", type=float, default=0.35)
    parser.add_argument("--min-green-reflectance", type=float, default=1200.0)
    parser.add_argument("--cld-ignore-threshold", type=float, default=50.0)
    parser.add_argument("--split-train", type=float, default=0.70)
    parser.add_argument("--split-val", type=float, default=0.15)
    parser.add_argument("--split-test", type=float, default=0.15)
    parser.add_argument("--split-unit", choices=["scene", "city_year"], default="scene")
    parser.add_argument("--outdir", default="outputs_cnn_dataset")
    return parser.parse_args()


def parse_month_day(text: str) -> tuple[int, int]:
    try:
        parsed = datetime.strptime(text, "%m-%d")
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Bad month-day '{text}'. Use MM-DD.") from exc
    return parsed.month, parsed.day


def build_windows(years: list[int], start_mmdd: str, end_mmdd: str) -> list[dict]:
    start_month, start_day = parse_month_day(start_mmdd)
    end_month, end_day = parse_month_day(end_mmdd)
    wraps_year = (end_month, end_day) < (start_month, start_day)

    windows = []
    for year in years:
        start = date(year, start_month, start_day)
        end_year = year + 1 if wraps_year else year
        end = date(end_year, end_month, end_day)
        windows.append(
            {
                "window_year": year,
                "start_date": start.isoformat(),
                "end_date": end.isoformat(),
            }
        )
    return windows


def load_city_bboxes(city_file: str | None) -> dict[str, list[float]]:
    bboxes = dict(CITY_BBOXES)
    if not city_file:
        return bboxes

    payload = json.loads(Path(city_file).read_text())
    if not isinstance(payload, dict):
        raise ValueError("City file must be a JSON object like {'city': [west, south, east, north]}.")

    for city, bbox in payload.items():
        if not isinstance(bbox, list) or len(bbox) != 4:
            raise ValueError(f"City '{city}' must map to a 4-element bbox list.")
        bboxes[city.lower()] = [float(x) for x in bbox]

    return bboxes


def resolve_cities(requested: list[str], available: dict[str, list[float]]) -> list[str]:
    if len(requested) == 1 and requested[0].lower() == "all":
        return sorted(available)

    cities = []
    missing = []
    for city in requested:
        key = city.lower()
        if key in available:
            cities.append(key)
        else:
            missing.append(city)

    if missing:
        raise ValueError(
            f"Unknown cities: {missing}. Available built-ins: {sorted(available)}. "
            "You can also provide --city-file custom_cities.json"
        )
    return cities


def resolve_years(years: list[int] | None, year_start: int | None, year_end: int | None) -> list[int]:
    if years:
        return sorted(set(int(year) for year in years))
    if year_start is None or year_end is None:
        raise ValueError("Provide either --years or both --year-start and --year-end.")
    if year_end < year_start:
        raise ValueError("--year-end must be >= --year-start")
    return list(range(year_start, year_end + 1))


def validate_args(args: argparse.Namespace) -> tuple[list[str], SplitConfig, PatchConfig, float | None]:
    split_total = args.split_train + args.split_val + args.split_test
    if abs(split_total - 1.0) > 1e-8:
        raise ValueError("split-train + split-val + split-test must sum to 1.0")

    bands = list(dict.fromkeys(args.bands))
    if "B03" not in bands or "B11" not in bands:
        raise ValueError("--bands must include B03 and B11.")
    if args.patch_size <= 0 or args.stride <= 0:
        raise ValueError("--patch-size and --stride must both be positive.")

    split_cfg = SplitConfig(
        train=args.split_train,
        val=args.split_val,
        test=args.split_test,
        unit=args.split_unit,
    )
    patch_cfg = PatchConfig(
        size=args.patch_size,
        stride=args.stride,
        min_labeled_ratio=args.min_labeled_ratio,
        min_clear_ratio=args.min_clear_ratio,
        background_keep_prob=args.background_keep_prob,
        min_snow_ratio_positive=args.min_snow_ratio_positive,
    )
    cld_ignore_threshold = None if args.cld_ignore_threshold < 0 else args.cld_ignore_threshold
    return bands, split_cfg, patch_cfg, cld_ignore_threshold


def open_catalog() -> Client:
    return Client.open(EARTH_SEARCH_URL)


def bbox_to_polygon(bbox: list[float]) -> dict:
    west, south, east, north = bbox
    return {
        "type": "Polygon",
        "coordinates": [[[west, south], [east, south], [east, north], [west, north], [west, south]]],
    }


def search_items(
    catalog: Client,
    bbox: list[float],
    start_date: str,
    end_date: str,
    max_items: int | None,
    max_cloud: float,
) -> list:
    search = catalog.search(
        collections=["sentinel-2-l2a"],
        intersects=bbox_to_polygon(bbox),
        datetime=f"{start_date}/{end_date}",
    )
    items = [
        item for item in search.items()
        if item.properties.get("eo:cloud_cover") is not None
        and item.properties.get("eo:cloud_cover", 999.0) <= max_cloud
    ]
    items.sort(key=lambda item: (item.properties.get("eo:cloud_cover", 999.0), item.datetime or datetime.min))
    if max_items and max_items > 0:
        items = items[:max_items]
    return items


def get_asset_href(item, asset_name: str, required: bool = True) -> str | None:
    for key in ASSET_KEYS[asset_name]:
        asset = item.assets.get(key)
        if asset is not None:
            return asset.href
    if required:
        raise KeyError(
            f"Could not find asset '{asset_name}'. "
            f"Tried {ASSET_KEYS[asset_name]}; available keys: {list(item.assets.keys())}"
        )
    return None


def read_crop(
    href: str,
    bbox_lonlat: list[float],
    out_shape: tuple[int, int] | None = None,
    resampling: Resampling = Resampling.bilinear,
) -> np.ndarray:
    with rasterio.open(href) as src:
        crop_bounds = transform_bounds("EPSG:4326", src.crs, *bbox_lonlat, densify_pts=21)
        window = from_bounds(*crop_bounds, transform=src.transform).round_offsets().round_lengths()
        if window.width <= 0 or window.height <= 0:
            raise ValueError("Crop window came out empty. Check the bbox.")

        read_kwargs = {"window": window, "boundless": True, "masked": True}
        if out_shape is not None:
            read_kwargs["out_shape"] = (1, out_shape[0], out_shape[1])
            read_kwargs["resampling"] = resampling

        array = src.read(1, **read_kwargs)
        if np.ma.isMaskedArray(array):
            array = array.astype(np.float32).filled(np.nan)
        else:
            array = array.astype(np.float32)
        return np.asarray(array, dtype=np.float32)


def load_scene_arrays(item, bbox: list[float], bands: list[str]) -> dict[str, np.ndarray]:
    reference = read_crop(get_asset_href(item, "B03"), bbox)
    shape = reference.shape

    arrays = {"B03": reference}
    for band in bands:
        if band == "B03":
            continue
        arrays[band] = read_crop(
            get_asset_href(item, band),
            bbox,
            out_shape=shape,
            resampling=Resampling.bilinear,
        )

    arrays["SCL"] = read_crop(
        get_asset_href(item, "SCL"),
        bbox,
        out_shape=shape,
        resampling=Resampling.nearest,
    )

    cld_href = get_asset_href(item, "CLD", required=False)
    if cld_href is not None:
        arrays["CLD"] = read_crop(cld_href, bbox, out_shape=shape, resampling=Resampling.bilinear)

    snw_href = get_asset_href(item, "SNW", required=False)
    if snw_href is not None:
        arrays["SNW"] = read_crop(snw_href, bbox, out_shape=shape, resampling=Resampling.bilinear)

    return arrays


def build_pseudo_labels(
    arrays: dict[str, np.ndarray],
    bands: list[str],
    ndsi_threshold: float,
    min_green_reflectance: float,
    cld_ignore_threshold: float | None,
) -> dict[str, np.ndarray]:
    green = arrays["B03"]
    swir = arrays["B11"]
    scl = arrays["SCL"]

    valid = np.isfinite(scl)
    for band in bands:
        valid &= np.isfinite(arrays[band])

    scl_int = np.full(scl.shape, -9999, dtype=np.int16)
    scl_int[np.isfinite(scl)] = scl[np.isfinite(scl)].astype(np.int16)

    ndsi = np.full(green.shape, np.nan, dtype=np.float32)
    denom = green + swir
    good_ndsi = valid & (denom != 0)
    ndsi[good_ndsi] = (green[good_ndsi] - swir[good_ndsi]) / denom[good_ndsi]

    cloud_like = np.isin(scl_int, list(CLOUD_CLASSES | NODATA_CLASSES))
    water = scl_int == WATER_CLASS
    spectral_snow = valid & (green >= min_green_reflectance) & (ndsi >= ndsi_threshold)
    scl_snow = valid & (scl_int == SNOW_CLASS)

    ignore = ~valid | cloud_like | water
    if cld_ignore_threshold is not None and "CLD" in arrays:
        ignore |= np.isfinite(arrays["CLD"]) & (arrays["CLD"] >= cld_ignore_threshold)

    clear_land = valid & ~ignore
    positive = clear_land & (spectral_snow | scl_snow)

    label = np.full(green.shape, IGNORE_LABEL, dtype=np.uint8)
    label[clear_land & ~positive] = 0
    label[positive] = 1

    return {
        "label": label,
        "ndsi": ndsi.astype(np.float32),
        "clear_land": clear_land.astype(np.uint8),
    }


def build_image_stack(arrays: dict[str, np.ndarray], bands: list[str]) -> np.ndarray:
    return np.stack([arrays[band] for band in bands], axis=0).astype(np.float32) / 10000.0


def get_scene_datetime(item) -> str | None:
    if item.datetime is not None:
        return item.datetime.isoformat()
    return item.properties.get("datetime")


def get_scene_date(item) -> str | None:
    dt = get_scene_datetime(item)
    if dt is None:
        return None
    return pd.to_datetime(dt).date().isoformat()


def make_patch_starts(length: int, patch_size: int, stride: int) -> list[int]:
    if length <= patch_size:
        return [0]
    starts = list(range(0, length - patch_size + 1, stride))
    if starts[-1] != length - patch_size:
        starts.append(length - patch_size)
    return starts


def patch_metrics(label_patch: np.ndarray, clear_land_patch: np.ndarray) -> dict[str, float]:
    labeled = label_patch != IGNORE_LABEL
    labeled_ratio = float(np.mean(labeled))
    clear_ratio = float(np.mean(clear_land_patch > 0))
    positive_ratio = float(np.mean(label_patch == 1))
    positive_among_labeled = float(np.mean(label_patch[labeled] == 1)) if np.any(labeled) else 0.0
    return {
        "labeled_ratio": labeled_ratio,
        "clear_ratio": clear_ratio,
        "positive_ratio": positive_ratio,
        "positive_among_labeled": positive_among_labeled,
    }


def hash_to_unit(text: str) -> float:
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:15]
    return int(digest, 16) / float(16**15 - 1)


def choose_split(city: str, window_year: int, item_id: str, split_cfg: SplitConfig) -> tuple[str, str]:
    if split_cfg.unit == "scene":
        split_key = item_id
    elif split_cfg.unit == "city_year":
        split_key = f"{city}::{window_year}"
    else:
        raise ValueError(f"Unknown split unit: {split_cfg.unit}")

    draw = hash_to_unit(split_key)
    if draw < split_cfg.train:
        return "train", split_key
    if draw < split_cfg.train + split_cfg.val:
        return "val", split_key
    return "test", split_key


def keep_patch(metrics: dict[str, float], patch_cfg: PatchConfig, sample_id: str) -> bool:
    if metrics["labeled_ratio"] < patch_cfg.min_labeled_ratio:
        return False
    if metrics["clear_ratio"] < patch_cfg.min_clear_ratio:
        return False
    if metrics["positive_among_labeled"] >= patch_cfg.min_snow_ratio_positive:
        return True
    return hash_to_unit(sample_id) < patch_cfg.background_keep_prob


def save_sample(path: Path, image: np.ndarray, label: np.ndarray, ndsi: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, image=image.astype(np.float32), label=label.astype(np.uint8), ndsi=ndsi.astype(np.float32))


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def save_csv(path: Path, rows: list[dict], sort_columns: list[str]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame = frame.sort_values(sort_columns).reset_index(drop=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return frame


def compute_train_stats(samples_dir: Path, patch_df: pd.DataFrame, bands: list[str]) -> dict:
    if patch_df.empty:
        return {"band_names": bands, "train_patch_count": 0, "mean": None, "std": None, "pixel_counts": None}

    train_df = patch_df[patch_df["split"] == "train"].copy()
    if train_df.empty:
        return {"band_names": bands, "train_patch_count": 0, "mean": None, "std": None, "pixel_counts": None}

    sums = np.zeros(len(bands), dtype=np.float64)
    sums_sq = np.zeros(len(bands), dtype=np.float64)
    counts = np.zeros(len(bands), dtype=np.float64)

    for rel_path in train_df["sample_relpath"].tolist():
        payload = np.load(samples_dir / rel_path)
        image = payload["image"].astype(np.float64)
        label = payload["label"]
        valid = label != IGNORE_LABEL

        for index in range(image.shape[0]):
            values = image[index][valid]
            if values.size == 0:
                continue
            sums[index] += values.sum()
            sums_sq[index] += np.square(values).sum()
            counts[index] += values.size

    mean = np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0)
    variance = np.divide(sums_sq, counts, out=np.zeros_like(sums_sq), where=counts > 0) - np.square(mean)
    variance = np.maximum(variance, 0.0)
    std = np.sqrt(variance)

    return {
        "band_names": bands,
        "train_patch_count": int(len(train_df)),
        "mean": [float(value) for value in mean],
        "std": [float(value) for value in std],
        "pixel_counts": [int(value) for value in counts],
    }


def process_scene(
    item,
    city: str,
    bbox: list[float],
    window_year: int,
    bands: list[str],
    split_cfg: SplitConfig,
    patch_cfg: PatchConfig,
    args: argparse.Namespace,
    cld_ignore_threshold: float | None,
    samples_dir: Path,
) -> tuple[dict | None, list[dict]]:
    split, split_key = choose_split(city, window_year, item.id, split_cfg)
    dt = get_scene_datetime(item)
    dt_day = get_scene_date(item)

    arrays = load_scene_arrays(item, bbox, bands)
    pseudo = build_pseudo_labels(
        arrays=arrays,
        bands=bands,
        ndsi_threshold=args.ndsi_threshold,
        min_green_reflectance=args.min_green_reflectance,
        cld_ignore_threshold=cld_ignore_threshold,
    )
    image = build_image_stack(arrays, bands)
    label = pseudo["label"]
    ndsi = pseudo["ndsi"]
    clear_land = pseudo["clear_land"]

    if image.shape[1] < patch_cfg.size or image.shape[2] < patch_cfg.size:
        print(
            f"[skip] {city} | {item.id} is smaller than one patch "
            f"({image.shape[1]}x{image.shape[2]} < {patch_cfg.size})"
        )
        return None, []

    scene_stats = patch_metrics(label, clear_land)
    scene_row = {
        "city": city,
        "window_year": window_year,
        "date": dt_day,
        "datetime": dt,
        "item_id": item.id,
        "split": split,
        "split_key": split_key,
        "shape_h": int(image.shape[1]),
        "shape_w": int(image.shape[2]),
        "cloud_cover_tile_pct": item.properties.get("eo:cloud_cover"),
        "snow_cover_tile_pct": item.properties.get("eo:snow_cover"),
        "sentinel_tile_id": item.properties.get("s2:mgrs_tile") or item.properties.get("mgrs:tile"),
        "scene_labeled_ratio": scene_stats["labeled_ratio"],
        "scene_clear_ratio": scene_stats["clear_ratio"],
        "scene_positive_ratio": scene_stats["positive_ratio"],
        "scene_positive_among_labeled": scene_stats["positive_among_labeled"],
    }

    patch_rows = []
    kept_count = 0
    y_starts = make_patch_starts(image.shape[1], patch_cfg.size, patch_cfg.stride)
    x_starts = make_patch_starts(image.shape[2], patch_cfg.size, patch_cfg.stride)

    for y0 in y_starts:
        for x0 in x_starts:
            y1 = y0 + patch_cfg.size
            x1 = x0 + patch_cfg.size

            image_patch = image[:, y0:y1, x0:x1]
            label_patch = label[y0:y1, x0:x1]
            ndsi_patch = ndsi[y0:y1, x0:x1]
            clear_patch = clear_land[y0:y1, x0:x1]

            if image_patch.shape[1] != patch_cfg.size or image_patch.shape[2] != patch_cfg.size:
                continue

            metrics = patch_metrics(label_patch, clear_patch)
            sample_id = f"{city}__wy{window_year}__{item.id}__y{y0:04d}_x{x0:04d}"
            if not keep_patch(metrics, patch_cfg, sample_id):
                continue

            sample_relpath = Path(split) / city / f"{sample_id}.npz"
            save_sample(samples_dir / sample_relpath, image_patch, label_patch, ndsi_patch)

            patch_rows.append(
                {
                    "sample_id": sample_id,
                    "sample_relpath": str(sample_relpath),
                    "city": city,
                    "window_year": window_year,
                    "date": dt_day,
                    "datetime": dt,
                    "item_id": item.id,
                    "split": split,
                    "split_key": split_key,
                    "patch_y0": y0,
                    "patch_x0": x0,
                    "patch_size": patch_cfg.size,
                    "bands": ",".join(bands),
                    "cloud_cover_tile_pct": item.properties.get("eo:cloud_cover"),
                    "snow_cover_tile_pct": item.properties.get("eo:snow_cover"),
                    "sentinel_tile_id": item.properties.get("s2:mgrs_tile") or item.properties.get("mgrs:tile"),
                    **metrics,
                }
            )
            kept_count += 1

    print(
        f"[ok] {city} | {item.id} | split={split} | "
        f"shape={image.shape[1]}x{image.shape[2]} | kept={kept_count} | "
        f"snow_ratio={scene_stats['positive_among_labeled']:.4f}"
    )
    return scene_row, patch_rows


def write_split_files(manifests_dir: Path, outdir: Path, patch_df: pd.DataFrame) -> dict[str, str]:
    split_files = {}
    for split_name in ["train", "val", "test"]:
        split_df = patch_df[patch_df["split"] == split_name].copy() if not patch_df.empty else pd.DataFrame()
        split_path = manifests_dir / f"{split_name}.csv"
        split_df.to_csv(split_path, index=False)
        split_files[split_name] = str(split_path.relative_to(outdir))
    return split_files


def build_summary(
    args: argparse.Namespace,
    bands: list[str],
    cities: list[str],
    years: list[int],
    split_cfg: SplitConfig,
    patch_cfg: PatchConfig,
    cld_ignore_threshold: float | None,
    split_files: dict[str, str],
    window_df: pd.DataFrame,
    scene_df: pd.DataFrame,
    patch_df: pd.DataFrame,
) -> dict:
    return {
        "bands": bands,
        "cities": cities,
        "years": years,
        "season_window": {
            "start_month_day": args.start_month_day,
            "end_month_day": args.end_month_day,
        },
        "search": {
            "max_items_per_window": args.max_items_per_window,
            "max_cloud": args.max_cloud,
        },
        "patching": {
            "patch_size": patch_cfg.size,
            "stride": patch_cfg.stride,
            "min_labeled_ratio": patch_cfg.min_labeled_ratio,
            "min_clear_ratio": patch_cfg.min_clear_ratio,
            "background_keep_prob": patch_cfg.background_keep_prob,
            "min_snow_ratio_positive": patch_cfg.min_snow_ratio_positive,
        },
        "pseudo_labels": {
            "ndsi_threshold": args.ndsi_threshold,
            "min_green_reflectance": args.min_green_reflectance,
            "cld_ignore_threshold": cld_ignore_threshold,
            "ignore_label": IGNORE_LABEL,
        },
        "splits": {
            "unit": split_cfg.unit,
            "train": split_cfg.train,
            "val": split_cfg.val,
            "test": split_cfg.test,
            "files": split_files,
        },
        "counts": {
            "window_rows": int(len(window_df)),
            "scene_rows": int(len(scene_df)),
            "patch_rows": int(len(patch_df)),
            "train_patches": int((patch_df["split"] == "train").sum()) if not patch_df.empty else 0,
            "val_patches": int((patch_df["split"] == "val").sum()) if not patch_df.empty else 0,
            "test_patches": int((patch_df["split"] == "test").sum()) if not patch_df.empty else 0,
        },
        "sample_format": {
            "relative_root": "samples",
            "file_type": "npz",
            "arrays": {
                "image": {
                    "shape": ["C", "H", "W"],
                    "dtype": "float32",
                    "scale": "surface_reflectance_div_10000",
                },
                "label": {
                    "shape": ["H", "W"],
                    "dtype": "uint8",
                    "values": {
                        "0": "non_snow_clear_land",
                        "1": "snow",
                        str(IGNORE_LABEL): "ignore_cloud_water_invalid",
                    },
                },
                "ndsi": {
                    "shape": ["H", "W"],
                    "dtype": "float32",
                },
            },
        },
    }


def print_outputs(outdir: Path, manifests_dir: Path, metadata_dir: Path) -> None:
    print(f"\nSaved dataset to: {outdir.resolve()}")
    print("Wrote:")
    print(f"  {manifests_dir / 'window_manifest.csv'}")
    print(f"  {manifests_dir / 'scene_manifest.csv'}")
    print(f"  {manifests_dir / 'patch_manifest.csv'}")
    print(f"  {manifests_dir / 'train.csv'}")
    print(f"  {manifests_dir / 'val.csv'}")
    print(f"  {manifests_dir / 'test.csv'}")
    print(f"  {metadata_dir / 'train_channel_stats.json'}")
    print(f"  {metadata_dir / 'dataset_config.json'}")
    print("Each .npz sample contains image [C,H,W], label [H,W], ndsi [H,W].")


def main() -> None:
    args = parse_args()
    bands, split_cfg, patch_cfg, cld_ignore_threshold = validate_args(args)

    available_cities = load_city_bboxes(args.city_file)
    selected_cities = resolve_cities(args.cities, available_cities)
    years = resolve_years(args.years, args.year_start, args.year_end)
    windows = build_windows(years, args.start_month_day, args.end_month_day)

    outdir = Path(args.outdir)
    samples_dir = outdir / "samples"
    manifests_dir = outdir / "manifests"
    metadata_dir = outdir / "metadata"
    samples_dir.mkdir(parents=True, exist_ok=True)
    manifests_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    catalog = open_catalog()
    window_rows = []
    scene_rows = []
    patch_rows = []

    for city in selected_cities:
        bbox = available_cities[city]
        for window in windows:
            window_year = int(window["window_year"])
            start_date = str(window["start_date"])
            end_date = str(window["end_date"])
            print(f"\n=== {city} | {start_date} -> {end_date} ===")

            try:
                items = search_items(catalog, bbox, start_date, end_date, args.max_items_per_window, args.max_cloud)
            except Exception as exc:
                print(f"[warn] search failed for {city} {window_year}: {exc}")
                window_rows.append(
                    {
                        "city": city,
                        "window_year": window_year,
                        "start_date": start_date,
                        "end_date": end_date,
                        "items_found": 0,
                        "search_error": str(exc),
                    }
                )
                continue

            window_rows.append(
                {
                    "city": city,
                    "window_year": window_year,
                    "start_date": start_date,
                    "end_date": end_date,
                    "items_found": len(items),
                    "search_error": None,
                }
            )

            for item in items:
                try:
                    scene_row, scene_patch_rows = process_scene(
                        item=item,
                        city=city,
                        bbox=bbox,
                        window_year=window_year,
                        bands=bands,
                        split_cfg=split_cfg,
                        patch_cfg=patch_cfg,
                        args=args,
                        cld_ignore_threshold=cld_ignore_threshold,
                        samples_dir=samples_dir,
                    )
                except Exception as exc:
                    print(f"[warn] failed to load {item.id} for {city} {window_year}: {exc}")
                    continue

                if scene_row is not None:
                    scene_rows.append(scene_row)
                patch_rows.extend(scene_patch_rows)

    window_df = save_csv(manifests_dir / "window_manifest.csv", window_rows, ["city", "window_year"])
    scene_df = save_csv(
        manifests_dir / "scene_manifest.csv",
        scene_rows,
        ["city", "window_year", "datetime", "item_id"],
    )
    patch_df = save_csv(
        manifests_dir / "patch_manifest.csv",
        patch_rows,
        ["split", "city", "window_year", "datetime", "item_id", "patch_y0", "patch_x0"],
    )

    split_files = write_split_files(manifests_dir, outdir, patch_df)

    train_stats = compute_train_stats(samples_dir, patch_df, bands)
    save_json(metadata_dir / "train_channel_stats.json", train_stats)

    save_json(
        metadata_dir / "label_map.json",
        {
            "0": "non_snow_clear_land",
            "1": "snow",
            str(IGNORE_LABEL): "ignore_cloud_water_invalid",
        },
    )

    summary = build_summary(
        args=args,
        bands=bands,
        cities=selected_cities,
        years=years,
        split_cfg=split_cfg,
        patch_cfg=patch_cfg,
        cld_ignore_threshold=cld_ignore_threshold,
        split_files=split_files,
        window_df=window_df,
        scene_df=scene_df,
        patch_df=patch_df,
    )
    save_json(metadata_dir / "dataset_config.json", summary)
    print_outputs(outdir, manifests_dir, metadata_dir)


if __name__ == "__main__":
    main()
