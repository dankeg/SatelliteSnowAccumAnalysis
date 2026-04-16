from __future__ import annotations

import hashlib
import json
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from pystac_client import Client
from rasterio.enums import Resampling
from rasterio.windows import from_bounds
from rasterio.warp import transform_bounds


EARTH_SEARCH_URL = "https://earth-search.aws.element84.com/v1"
COLLECTION = "sentinel-2-l2a"
IGNORE_LABEL = 255
BANDS = ["B02", "B03", "B04", "B08", "B11", "B12"]

CITY_BBOXES = {
    "boston": [-71.1912, 42.2279, -70.9860, 42.3995],
    "new_york": [-74.2591, 40.4774, -73.7004, 40.9176],
    "buffalo": [-78.9370, 42.8261, -78.7382, 42.9585],
    "chicago": [-87.9401, 41.6445, -87.5237, 42.0230],
}

CITIES = ["boston", "new_york", "buffalo", "chicago"]
YEARS = list(range(2015, 2025))
SEASON_START = "11-15"
SEASON_END = "02-28"
MAX_ITEMS_PER_WINDOW = 5
MAX_CLOUD = 25
PATCH_SIZE = 256
STRIDE = 256
MIN_LABELED_RATIO = 0.50
MIN_CLEAR_RATIO = 0.50
BACKGROUND_KEEP_PROB = 0.10
MIN_SNOW_RATIO_POSITIVE = 0.01
OUTDIR = Path("snow_dataset_small")

TRAIN_SPLIT = 0.70
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15

ASSETS = {
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

NO_DATA = {0, 1}
SHADOW = {2, 3}
CLOUD = {8, 9, 10}
WATER = 6
SNOW = 11

SNW_STRONG = 35.0
SNW_SUPPORT = 20.0
CLD_STRONG = 65.0
NDSI_STRONG = 0.42
NDSI_RELAXED = 0.25
MIN_GREEN = 0.10
MIN_RGB_MEAN = 0.12
MIN_RGB_MEAN_RELAXED = 0.10
MIN_NIR = 0.10
MAX_SWIR = 0.18
WATER_NDWI = 0.15
MAX_WATER_NIR = 0.12
MAX_WATER_SWIR = 0.10


def mmdd(text: str) -> tuple[int, int]:
    dt = datetime.strptime(text, "%m-%d")
    return dt.month, dt.day


def build_windows(years: list[int]) -> list[dict[str, str | int]]:
    start_month, start_day = mmdd(SEASON_START)
    end_month, end_day = mmdd(SEASON_END)
    wraps = (end_month, end_day) < (start_month, start_day)

    windows = []
    for year in years:
        start = date(year, start_month, start_day)
        end = date(year + 1 if wraps else year, end_month, end_day)
        windows.append(
            {
                "window_year": year,
                "start_date": start.isoformat(),
                "end_date": end.isoformat(),
            }
        )
    return windows


def bbox_polygon(bbox: list[float]) -> dict:
    west, south, east, north = bbox
    return {
        "type": "Polygon",
        "coordinates": [[[west, south], [east, south], [east, north], [west, north], [west, south]]],
    }


def open_catalog() -> Client:
    return Client.open(EARTH_SEARCH_URL)


def search_items(catalog: Client, bbox: list[float], start_date: str, end_date: str) -> list:
    search = catalog.search(
        collections=[COLLECTION],
        intersects=bbox_polygon(bbox),
        datetime=f"{start_date}/{end_date}",
    )

    items = []
    for item in search.items():
        cloud = item.properties.get("eo:cloud_cover")
        if cloud is None or cloud > MAX_CLOUD:
            continue
        items.append(item)

    items.sort(key=lambda item: (item.properties.get("eo:cloud_cover", 999), item.datetime or datetime.min))
    return items[:MAX_ITEMS_PER_WINDOW]


def asset_href(item, name: str, required: bool = True) -> str | None:
    for key in ASSETS[name]:
        asset = item.assets.get(key)
        if asset is not None:
            return asset.href
    if required:
        raise KeyError(f"Missing {name} asset. Available assets: {list(item.assets.keys())}")
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
        array = src.read(
            1,
            window=window,
            boundless=True,
            masked=True,
            out_shape=(1, out_shape[0], out_shape[1]) if out_shape else None,
            resampling=resampling if out_shape else Resampling.nearest,
        )
    if np.ma.isMaskedArray(array):
        array = array.astype(np.float32).filled(np.nan)
    return np.asarray(array, dtype=np.float32)


def load_scene(item, bbox: list[float]) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {}

    green = read_crop(asset_href(item, "B03"), bbox)
    shape = green.shape
    arrays["B03"] = green

    for band in BANDS:
        if band == "B03":
            continue
        arrays[band] = read_crop(asset_href(item, band), bbox, out_shape=shape, resampling=Resampling.bilinear)

    arrays["SCL"] = read_crop(asset_href(item, "SCL"), bbox, out_shape=shape, resampling=Resampling.nearest)

    cld_href = asset_href(item, "CLD", required=False)
    if cld_href:
        arrays["CLD"] = read_crop(cld_href, bbox, out_shape=shape, resampling=Resampling.bilinear)

    snw_href = asset_href(item, "SNW", required=False)
    if snw_href:
        arrays["SNW"] = read_crop(snw_href, bbox, out_shape=shape, resampling=Resampling.bilinear)

    return arrays


def norm_diff(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    out = np.full(a.shape, np.nan, dtype=np.float32)
    denom = a + b
    mask = np.isfinite(a) & np.isfinite(b) & (denom != 0)
    out[mask] = (a[mask] - b[mask]) / denom[mask]
    return out


def build_labels(arrays: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    blue = arrays["B02"] / 10000.0
    green = arrays["B03"] / 10000.0
    red = arrays["B04"] / 10000.0
    nir = arrays["B08"] / 10000.0
    swir1 = arrays["B11"] / 10000.0
    scl = arrays["SCL"]

    valid = np.isfinite(scl)
    for band in BANDS:
        valid &= np.isfinite(arrays[band])

    scl_int = np.full(scl.shape, -9999, dtype=np.int16)
    scl_int[np.isfinite(scl)] = scl[np.isfinite(scl)].astype(np.int16)

    snw = arrays.get("SNW")
    cld = arrays.get("CLD")
    if snw is None:
        snw = np.full(scl.shape, np.nan, dtype=np.float32)
    if cld is None:
        cld = np.full(scl.shape, np.nan, dtype=np.float32)

    ndsi = norm_diff(green, swir1)
    ndwi = norm_diff(green, nir)
    rgb_mean = (blue + green + red) / 3.0

    water_like = (scl_int == WATER) | (
        np.isfinite(ndwi)
        & (ndwi > WATER_NDWI)
        & (nir < MAX_WATER_NIR)
        & (swir1 < MAX_WATER_SWIR)
    )

    strong_snow = (
        (scl_int == SNOW)
        | (np.isfinite(snw) & (snw >= SNW_STRONG))
        | (
            np.isfinite(ndsi)
            & (ndsi >= NDSI_STRONG)
            & (green >= MIN_GREEN)
            & (rgb_mean >= MIN_RGB_MEAN)
            & (nir >= MIN_NIR)
            & (swir1 <= MAX_SWIR)
        )
    )

    supported_snow = (
        np.isfinite(ndsi)
        & (ndsi >= NDSI_RELAXED)
        & (rgb_mean >= MIN_RGB_MEAN_RELAXED)
        & (
            (np.isfinite(snw) & (snw >= SNW_SUPPORT))
            | (scl_int == SNOW)
            | (np.isfinite(cld) & (cld < CLD_STRONG))
            | (~np.isin(scl_int, list(CLOUD)))
        )
    )

    snow = valid & ~water_like & (strong_snow | supported_snow)

    ignore = (
        ~valid
        | np.isin(scl_int, list(NO_DATA | SHADOW))
        | water_like
        | (((np.isin(scl_int, list(CLOUD))) | (np.isfinite(cld) & (cld >= CLD_STRONG))) & ~snow)
    )

    label = np.full(scl.shape, IGNORE_LABEL, dtype=np.uint8)
    label[valid & ~ignore] = 0
    label[snow & ~ignore] = 1

    clear_land = (label != IGNORE_LABEL).astype(np.uint8)
    return {
        "label": label,
        "ndsi": ndsi.astype(np.float32),
        "clear_land": clear_land,
        "snw": snw.astype(np.float32),
        "cld": cld.astype(np.float32),
    }


def build_image(arrays: dict[str, np.ndarray]) -> np.ndarray:
    return np.stack([arrays[band] for band in BANDS], axis=0).astype(np.float32) / 10000.0


def scene_datetime(item) -> str | None:
    if item.datetime is not None:
        return item.datetime.isoformat()
    return item.properties.get("datetime")


def scene_date(item) -> str | None:
    dt = scene_datetime(item)
    return pd.to_datetime(dt).date().isoformat() if dt else None


def patch_starts(length: int, patch_size: int, stride: int) -> list[int]:
    if length <= patch_size:
        return [0]
    starts = list(range(0, length - patch_size + 1, stride))
    if starts[-1] != length - patch_size:
        starts.append(length - patch_size)
    return starts


def patch_stats(label_patch: np.ndarray, clear_land_patch: np.ndarray) -> dict[str, float]:
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


def hash_unit(text: str) -> float:
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:15]
    return int(digest, 16) / float(16**15 - 1)


def choose_split(item_id: str) -> str:
    draw = hash_unit(item_id)
    if draw < TRAIN_SPLIT:
        return "train"
    if draw < TRAIN_SPLIT + VAL_SPLIT:
        return "val"
    return "test"


def keep_patch(stats: dict[str, float], sample_id: str) -> bool:
    if stats["labeled_ratio"] < MIN_LABELED_RATIO:
        return False
    if stats["clear_ratio"] < MIN_CLEAR_RATIO:
        return False
    if stats["positive_among_labeled"] >= MIN_SNOW_RATIO_POSITIVE:
        return True
    return hash_unit(sample_id) < BACKGROUND_KEEP_PROB


def save_sample(path: Path, image: np.ndarray, label: np.ndarray, ndsi: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, image=image.astype(np.float32), label=label.astype(np.uint8), ndsi=ndsi.astype(np.float32))


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def save_csv(path: Path, rows: list[dict], sort_by: list[str]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(sort_by).reset_index(drop=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df


def compute_train_stats(samples_dir: Path, patch_df: pd.DataFrame) -> dict:
    train_df = patch_df[patch_df["split"] == "train"].copy() if not patch_df.empty else pd.DataFrame()
    if train_df.empty:
        return {"band_names": BANDS, "train_patch_count": 0, "mean": None, "std": None, "pixel_counts": None}

    sums = np.zeros(len(BANDS), dtype=np.float64)
    sums_sq = np.zeros(len(BANDS), dtype=np.float64)
    counts = np.zeros(len(BANDS), dtype=np.float64)

    for relpath in train_df["sample_relpath"].tolist():
        payload = np.load(samples_dir / relpath)
        image = payload["image"].astype(np.float64)
        label = payload["label"]
        valid = label != IGNORE_LABEL

        for i in range(len(BANDS)):
            values = image[i][valid]
            if values.size == 0:
                continue
            sums[i] += values.sum()
            sums_sq[i] += np.square(values).sum()
            counts[i] += values.size

    mean = np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0)
    var = np.divide(sums_sq, counts, out=np.zeros_like(sums_sq), where=counts > 0) - np.square(mean)
    var = np.maximum(var, 0.0)
    std = np.sqrt(var)

    return {
        "band_names": BANDS,
        "train_patch_count": int(len(train_df)),
        "mean": [float(x) for x in mean],
        "std": [float(x) for x in std],
        "pixel_counts": [int(x) for x in counts],
    }


def process_scene(item, city: str, bbox: list[float], window_year: int, samples_dir: Path) -> tuple[dict | None, list[dict]]:
    arrays = load_scene(item, bbox)
    labels = build_labels(arrays)
    image = build_image(arrays)
    label = labels["label"]
    ndsi = labels["ndsi"]
    clear_land = labels["clear_land"]

    if image.shape[1] < PATCH_SIZE or image.shape[2] < PATCH_SIZE:
        print(f"[skip] {city} | {item.id} is smaller than one patch")
        return None, []

    split = choose_split(item.id)
    dt = scene_datetime(item)
    dt_day = scene_date(item)

    scene_stats = patch_stats(label, clear_land)
    scene_row = {
        "city": city,
        "window_year": window_year,
        "date": dt_day,
        "datetime": dt,
        "item_id": item.id,
        "split": split,
        "shape_h": int(image.shape[1]),
        "shape_w": int(image.shape[2]),
        "cloud_cover_tile_pct": item.properties.get("eo:cloud_cover"),
        "snow_cover_tile_pct": item.properties.get("eo:snow_cover"),
        "scene_labeled_ratio": scene_stats["labeled_ratio"],
        "scene_clear_ratio": scene_stats["clear_ratio"],
        "scene_positive_ratio": scene_stats["positive_ratio"],
        "scene_positive_among_labeled": scene_stats["positive_among_labeled"],
    }

    patch_rows: list[dict] = []
    kept = 0
    for y0 in patch_starts(image.shape[1], PATCH_SIZE, STRIDE):
        for x0 in patch_starts(image.shape[2], PATCH_SIZE, STRIDE):
            y1 = y0 + PATCH_SIZE
            x1 = x0 + PATCH_SIZE

            image_patch = image[:, y0:y1, x0:x1]
            label_patch = label[y0:y1, x0:x1]
            ndsi_patch = ndsi[y0:y1, x0:x1]
            clear_patch = clear_land[y0:y1, x0:x1]

            if image_patch.shape[1] != PATCH_SIZE or image_patch.shape[2] != PATCH_SIZE:
                continue

            stats = patch_stats(label_patch, clear_patch)
            sample_id = f"{city}__wy{window_year}__{item.id}__y{y0:04d}_x{x0:04d}"
            if not keep_patch(stats, sample_id):
                continue

            relpath = Path(split) / city / f"{sample_id}.npz"
            save_sample(samples_dir / relpath, image_patch, label_patch, ndsi_patch)

            patch_rows.append(
                {
                    "sample_id": sample_id,
                    "sample_relpath": str(relpath),
                    "city": city,
                    "window_year": window_year,
                    "date": dt_day,
                    "datetime": dt,
                    "item_id": item.id,
                    "split": split,
                    "patch_y0": y0,
                    "patch_x0": x0,
                    "patch_size": PATCH_SIZE,
                    "bands": ",".join(BANDS),
                    "cloud_cover_tile_pct": item.properties.get("eo:cloud_cover"),
                    "snow_cover_tile_pct": item.properties.get("eo:snow_cover"),
                    **stats,
                }
            )
            kept += 1

    print(
        f"[ok] {city} | {item.id} | split={split} | kept={kept} | "
        f"snow_ratio={scene_stats['positive_among_labeled']:.4f}"
    )
    return scene_row, patch_rows


def write_split_csvs(manifests_dir: Path, patch_df: pd.DataFrame) -> dict[str, str]:
    files = {}
    for split in ["train", "val", "test"]:
        split_df = patch_df[patch_df["split"] == split].copy() if not patch_df.empty else pd.DataFrame()
        path = manifests_dir / f"{split}.csv"
        split_df.to_csv(path, index=False)
        files[split] = str(path)
    return files


def main() -> None:
    outdir = OUTDIR
    samples_dir = outdir / "samples"
    manifests_dir = outdir / "manifests"
    metadata_dir = outdir / "metadata"
    samples_dir.mkdir(parents=True, exist_ok=True)
    manifests_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    catalog = open_catalog()
    windows = build_windows(YEARS)

    window_rows: list[dict] = []
    scene_rows: list[dict] = []
    patch_rows: list[dict] = []

    for city in CITIES:
        bbox = CITY_BBOXES[city]
        for window in windows:
            year = int(window["window_year"])
            start_date = str(window["start_date"])
            end_date = str(window["end_date"])
            print(f"\n=== {city} | {start_date} -> {end_date} ===")

            try:
                items = search_items(catalog, bbox, start_date, end_date)
            except Exception as exc:
                print(f"[warn] search failed for {city} {year}: {exc}")
                window_rows.append(
                    {
                        "city": city,
                        "window_year": year,
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
                    "window_year": year,
                    "start_date": start_date,
                    "end_date": end_date,
                    "items_found": len(items),
                    "search_error": None,
                }
            )

            for item in items:
                try:
                    scene_row, scene_patch_rows = process_scene(item, city, bbox, year, samples_dir)
                except Exception as exc:
                    print(f"[warn] failed to load {item.id} for {city} {year}: {exc}")
                    continue

                if scene_row is not None:
                    scene_rows.append(scene_row)
                patch_rows.extend(scene_patch_rows)

    window_df = save_csv(manifests_dir / "window_manifest.csv", window_rows, ["city", "window_year"])
    scene_df = save_csv(manifests_dir / "scene_manifest.csv", scene_rows, ["city", "window_year", "datetime", "item_id"])
    patch_df = save_csv(
        manifests_dir / "patch_manifest.csv",
        patch_rows,
        ["split", "city", "window_year", "datetime", "item_id", "patch_y0", "patch_x0"],
    )

    split_files = write_split_csvs(manifests_dir, patch_df)
    train_stats = compute_train_stats(samples_dir, patch_df)

    save_json(metadata_dir / "train_channel_stats.json", train_stats)
    save_json(metadata_dir / "label_map.json", {"0": "non_snow_clear_land", "1": "snow", str(IGNORE_LABEL): "ignore"})
    save_json(
        metadata_dir / "dataset_config.json",
        {
            "cities": CITIES,
            "years": YEARS,
            "season_start": SEASON_START,
            "season_end": SEASON_END,
            "max_items_per_window": MAX_ITEMS_PER_WINDOW,
            "max_cloud": MAX_CLOUD,
            "bands": BANDS,
            "patch_size": PATCH_SIZE,
            "stride": STRIDE,
            "min_labeled_ratio": MIN_LABELED_RATIO,
            "min_clear_ratio": MIN_CLEAR_RATIO,
            "background_keep_prob": BACKGROUND_KEEP_PROB,
            "min_snow_ratio_positive": MIN_SNOW_RATIO_POSITIVE,
            "split": {"train": TRAIN_SPLIT, "val": VAL_SPLIT, "test": TEST_SPLIT, "unit": "scene"},
            "labeling": {
                "snw_strong": SNW_STRONG,
                "snw_support": SNW_SUPPORT,
                "cld_strong": CLD_STRONG,
                "ndsi_strong": NDSI_STRONG,
                "ndsi_relaxed": NDSI_RELAXED,
                "min_green": MIN_GREEN,
                "min_rgb_mean": MIN_RGB_MEAN,
                "min_nir": MIN_NIR,
                "max_swir": MAX_SWIR,
                "water_ndwi": WATER_NDWI,
                "max_water_nir": MAX_WATER_NIR,
                "max_water_swir": MAX_WATER_SWIR,
            },
            "counts": {
                "windows": int(len(window_df)),
                "scenes": int(len(scene_df)),
                "patches": int(len(patch_df)),
                "train_patches": int((patch_df["split"] == "train").sum()) if not patch_df.empty else 0,
                "val_patches": int((patch_df["split"] == "val").sum()) if not patch_df.empty else 0,
                "test_patches": int((patch_df["split"] == "test").sum()) if not patch_df.empty else 0,
            },
            "split_files": split_files,
        },
    )

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


if __name__ == "__main__":
    main()