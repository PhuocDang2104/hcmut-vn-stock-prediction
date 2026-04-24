from __future__ import annotations

import io
import json
import os
import subprocess
import sys
from datetime import date
from pathlib import Path
from time import sleep
from typing import Any

import pandas as pd
import yfinance as yf

from vnstock.data.loaders import load_universe
from vnstock.utils.io import ensure_dir, save_table, write_json
from vnstock.utils.paths import path_for, repo_root, resolve_path
from vnstock.utils.schema import RAW_COLUMNS

_VENDOR_SENTINEL = "__VNSTOCK_PAYLOAD_START__"


def _effective_end_date(config: dict[str, Any]) -> str:
    configured = config.get("end_date")
    if configured:
        return str(configured)
    return date.today().isoformat()


def _run_vendor_python(script: str, payload: dict[str, Any]) -> str:
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env["PYTHONIOENCODING"] = "utf-8"
    result = subprocess.run(
        [sys.executable, "-X", "utf8", "-c", script],
        input=json.dumps(payload),
        text=True,
        capture_output=True,
        encoding="utf-8",
        cwd=str(repo_root()),
        env=env,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip() or "Vendor vnstock call failed.")
    return result.stdout


def _extract_vendor_payload(stdout: str) -> str:
    if _VENDOR_SENTINEL not in stdout:
        preview = stdout[:500]
        raise ValueError(f"Vendor output missing sentinel. Preview: {preview!r}")
    return stdout.split(_VENDOR_SENTINEL, 1)[1].strip()


def _normalize_history_frame(
    frame: pd.DataFrame,
    symbol: str,
    source: str,
    start_date: str,
    end_date: str,
    value_mode: str | None,
) -> pd.DataFrame:
    result = frame.copy()
    result["date"] = pd.to_datetime(result["time"]).dt.tz_localize(None).dt.normalize()
    result = result.loc[
        (result["date"] >= pd.Timestamp(start_date)) & (result["date"] <= pd.Timestamp(end_date))
    ].copy()
    result["symbol"] = symbol.upper()
    result["source"] = source.upper()

    if "value" not in result.columns:
        if value_mode == "estimate_close_x_volume":
            result["value"] = result["close"].astype(float) * result["volume"].astype(float)
        else:
            result["value"] = pd.NA

    ordered = result[
        ["symbol", "date", "open", "high", "low", "close", "volume", "value", "source"]
    ].copy()
    ordered["date"] = ordered["date"].dt.date.astype(str)
    return ordered


def _normalize_yfinance_frame(
    frame: pd.DataFrame,
    symbol: str,
    source: str,
    start_date: str,
    end_date: str,
    value_mode: str | None,
) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=RAW_COLUMNS)

    result = frame.copy()
    if isinstance(result.columns, pd.MultiIndex):
        result.columns = result.columns.get_level_values(0)
    result = result.reset_index()
    result.columns = [str(column).lower() for column in result.columns]
    if "date" not in result.columns:
        raise ValueError(f"YFinance payload missing date column for {symbol}")

    result["date"] = pd.to_datetime(result["date"]).dt.tz_localize(None).dt.normalize()
    result = result.loc[
        (result["date"] >= pd.Timestamp(start_date)) & (result["date"] <= pd.Timestamp(end_date))
    ].copy()
    result["symbol"] = symbol.upper()
    result["source"] = source.upper()

    if "value" not in result.columns:
        if value_mode == "estimate_close_x_volume":
            result["value"] = result["close"].astype(float) * result["volume"].astype(float)
        else:
            result["value"] = pd.NA

    ordered = result[["symbol", "date", "open", "high", "low", "close", "volume", "value", "source"]].copy()
    ordered["date"] = ordered["date"].dt.date.astype(str)
    return ordered


def resolve_symbol_source(config: dict[str, Any], symbol: str) -> str:
    overrides = {
        str(key).upper(): str(value).upper()
        for key, value in config.get("symbol_source_overrides", {}).items()
    }
    return overrides.get(symbol.upper(), str(config.get("source", "KBS")).upper())


def _fetch_vnstock_symbol_history(
    symbol: str,
    source: str,
    start_date: str,
    end_date: str,
    interval: str = "1D",
    value_mode: str | None = None,
) -> pd.DataFrame:
    vendor_script = """
import json
import sys
from vnstock import Quote

payload = json.load(sys.stdin)
quote = Quote(symbol=payload["symbol"], source=payload["source"], show_log=False)
history = quote.history(
    start=payload["start_date"],
    end=payload["end_date"],
    interval=payload["interval"],
    get_all=True,
)
print("__VNSTOCK_PAYLOAD_START__")
print(history.to_json(orient="table", date_format="iso"))
"""
    raw_output = _run_vendor_python(
        vendor_script,
        {
            "symbol": symbol,
            "source": source,
            "start_date": start_date,
            "end_date": end_date,
            "interval": interval,
        },
    )
    history = pd.read_json(io.StringIO(_extract_vendor_payload(raw_output)), orient="table")
    return _normalize_history_frame(history, symbol, source, start_date, end_date, value_mode)


def _fetch_yfinance_symbol_history(
    symbol: str,
    source: str,
    start_date: str,
    end_date: str,
    interval: str = "1D",
    value_mode: str | None = None,
) -> pd.DataFrame:
    interval_map = {"1D": "1d", "1d": "1d"}
    yf_interval = interval_map.get(interval, interval.lower())
    if yf_interval != "1d":
        raise ValueError(f"Unsupported yfinance interval for {symbol}: {interval}")

    inclusive_end = (pd.Timestamp(end_date) + pd.Timedelta(days=1)).date().isoformat()
    history = yf.download(
        symbol,
        start=start_date,
        end=inclusive_end,
        interval=yf_interval,
        auto_adjust=True,
        progress=False,
        threads=False,
    )
    if history.empty:
        raise ValueError(f"Empty yfinance history for {symbol}")
    return _normalize_yfinance_frame(history, symbol, source, start_date, end_date, value_mode)


def fetch_symbol_history(
    symbol: str,
    source: str,
    start_date: str,
    end_date: str,
    interval: str = "1D",
    value_mode: str | None = None,
) -> pd.DataFrame:
    normalized_source = source.upper()
    if normalized_source == "YF":
        return _fetch_yfinance_symbol_history(
            symbol=symbol,
            source=normalized_source,
            start_date=start_date,
            end_date=end_date,
            interval=interval,
            value_mode=value_mode,
        )
    return _fetch_vnstock_symbol_history(
        symbol=symbol,
        source=normalized_source,
        start_date=start_date,
        end_date=end_date,
        interval=interval,
        value_mode=value_mode,
    )


def build_foreign_symbol_metadata(config: dict[str, Any], universe: list[str]) -> dict[str, Path]:
    configured_rows = list(config.get("foreign_symbol_metadata", []))
    if not configured_rows:
        return {}

    universe_set = {symbol.upper() for symbol in universe}
    rows = [
        {
            **row,
            "symbol": str(row["symbol"]).upper(),
            "source": resolve_symbol_source(config, str(row["symbol"])),
        }
        for row in configured_rows
        if str(row.get("symbol", "")).upper() in universe_set
    ]
    if not rows:
        return {}

    frame = pd.DataFrame(rows).sort_values(["market", "symbol"]).reset_index(drop=True)
    output_path = save_table(
        frame,
        path_for("raw_reference_root") / "foreign_symbol_metadata.csv",
        encoding="utf-8-sig",
    )
    return {"foreign_symbol_metadata": output_path}


def fetch_universe_history(config: dict[str, Any]) -> tuple[dict[str, Path], pd.DataFrame]:
    raw_root = ensure_dir(path_for("raw_root"))
    raw_download = config.get("raw_download", {})
    universe_scope = raw_download.get("universe_scope", "core")
    universe_path = resolve_path(config["universes"][universe_scope])
    universe = load_universe(universe_path)

    start_date = str(config["start_date"])
    end_date = _effective_end_date(config)
    interval = str(config.get("interval", "1D"))
    value_mode = raw_download.get("value_mode")
    sleep_seconds = float(raw_download.get("sleep_seconds", 0.0))

    outputs: dict[str, Path] = {}
    summaries: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []

    for symbol in universe:
        try:
            source = resolve_symbol_source(config, symbol)
            frame = fetch_symbol_history(
                symbol=symbol,
                source=source,
                start_date=start_date,
                end_date=end_date,
                interval=interval,
                value_mode=value_mode,
            )
            output_path = save_table(frame, raw_root / f"{symbol}.csv")
            outputs[symbol] = output_path
            summaries.append(
                {
                    "symbol": symbol,
                    "rows": int(len(frame)),
                    "date_min": frame["date"].min() if not frame.empty else None,
                    "date_max": frame["date"].max() if not frame.empty else None,
                    "source": source,
                }
            )
            if sleep_seconds > 0:
                sleep(sleep_seconds)
        except Exception as exc:  # pragma: no cover - network variability
            errors.append({"symbol": symbol, "error": repr(exc)})

    summary_frame = pd.DataFrame(summaries)
    if not summary_frame.empty:
        summary_frame = summary_frame.sort_values("symbol").reset_index(drop=True)
    save_table(summary_frame, path_for("raw_reference_root") / "raw_symbol_summary.csv")
    if errors:
        write_json({"errors": errors}, path_for("raw_reference_root") / "raw_download_errors.json")
        raise RuntimeError(f"Failed to fetch {len(errors)} symbols: {errors}")

    return outputs, summary_frame


def build_trading_calendar(raw_panel: pd.DataFrame) -> pd.DataFrame:
    calendar = pd.DataFrame({"date": sorted(pd.to_datetime(raw_panel["date"]).unique())})
    calendar["is_trading_day"] = True
    calendar["weekday"] = calendar["date"].dt.day_name()
    calendar["year"] = calendar["date"].dt.year
    calendar["month"] = calendar["date"].dt.month
    calendar["year_month"] = calendar["date"].dt.to_period("M").astype(str)
    calendar["days_since_prev_session"] = calendar["date"].diff().dt.days.fillna(0).astype(int)
    return calendar


def fetch_reference_data(config: dict[str, Any], universe: list[str]) -> dict[str, Path]:
    listing_source = str(config.get("listing_source", "VCI"))
    industry_listing_source = str(config.get("industry_listing_source", "KBS"))
    local_universe = [symbol for symbol in universe if resolve_symbol_source(config, symbol) != "YF"]
    reference_root = ensure_dir(path_for("raw_reference_root"))
    vendor_script = """
import json
import sys
import pandas as pd
from vnstock import Listing

payload = json.load(sys.stdin)
listing = Listing(source=payload["listing_source"], show_log=False)
universe = set(payload["universe"])
exchange_frames = []
for exchange in ("HOSE", "HNX", "UPCOM"):
    frame = listing.symbols_by_exchange(exchange=exchange)
    subset = frame.loc[frame["symbol"].isin(universe)].copy()
    if not subset.empty:
        exchange_frames.append(subset)

exchange_map = pd.concat(exchange_frames, ignore_index=True) if exchange_frames else pd.DataFrame()
if not exchange_map.empty:
    exchange_map = exchange_map.drop_duplicates(subset=["symbol"]).sort_values("symbol").reset_index(drop=True)

industry_map = pd.DataFrame()
industry_source = None
industry_errors = []
tried_sources = []
for source in (payload.get("listing_source"), payload.get("industry_listing_source"), "KBS"):
    normalized = (source or "").upper()
    if not normalized or normalized in tried_sources:
        continue
    tried_sources.append(normalized)
    try:
        listing_for_industry = Listing(source=normalized, show_log=False)
        candidate = listing_for_industry.symbols_by_industries()
        candidate = candidate.loc[candidate["symbol"].isin(universe)].copy()
        if not candidate.empty:
            candidate["industry_source"] = normalized
            industry_map = candidate.sort_values("symbol").reset_index(drop=True)
            industry_source = normalized
            break
    except Exception as exc:
        industry_errors.append({"source": normalized, "error": repr(exc)})

print("__VNSTOCK_PAYLOAD_START__")
print(json.dumps({
    "exchange_map": None if exchange_map.empty else exchange_map.to_json(orient="table", date_format="iso", force_ascii=False),
    "industry_map": None if industry_map.empty else industry_map.to_json(orient="table", date_format="iso", force_ascii=False),
    "industry_source": industry_source,
    "industry_errors": industry_errors,
}, ensure_ascii=False))
"""
    output_paths: dict[str, Path] = {}
    if local_universe:
        raw_output = _run_vendor_python(
            vendor_script,
            {
                "listing_source": listing_source,
                "industry_listing_source": industry_listing_source,
                "universe": local_universe,
            },
        )
        vendor_payload = json.loads(_extract_vendor_payload(raw_output))
        exchange_map = (
            pd.read_json(io.StringIO(vendor_payload["exchange_map"]), orient="table")
            if vendor_payload.get("exchange_map")
            else pd.DataFrame()
        )
        industry_map = (
            pd.read_json(io.StringIO(vendor_payload["industry_map"]), orient="table")
            if vendor_payload.get("industry_map")
            else pd.DataFrame()
        )

        if not exchange_map.empty:
            output_paths["exchange_mapping"] = save_table(
                exchange_map,
                reference_root / "exchange_mapping.csv",
                encoding="utf-8-sig",
            )
        if not industry_map.empty:
            output_paths["industry_mapping"] = save_table(
                industry_map,
                reference_root / "industry_mapping.csv",
                encoding="utf-8-sig",
            )
        if vendor_payload.get("industry_errors"):
            output_paths["reference_fallbacks"] = write_json(
                vendor_payload["industry_errors"],
                reference_root / "reference_fetch_warnings.json",
            )

    output_paths.update(build_foreign_symbol_metadata(config, universe))
    return output_paths


def fetch_index_history(config: dict[str, Any]) -> dict[str, Path]:
    index_root = ensure_dir(path_for("raw_index_root"))
    raw_download = config.get("raw_download", {})
    indices = list(raw_download.get("include_indices", []))
    if not indices:
        return {}

    start_date = str(config["start_date"])
    end_date = _effective_end_date(config)
    source = str(config.get("source", "KBS")).upper()
    output_paths: dict[str, Path] = {}

    for symbol in indices:
        frame = fetch_symbol_history(
            symbol=symbol,
            source=source,
            start_date=start_date,
            end_date=end_date,
            interval=str(config.get("interval", "1D")),
            value_mode=raw_download.get("value_mode"),
        )
        output_paths[symbol] = save_table(frame, index_root / f"{symbol}.csv")

    return output_paths


def bootstrap_raw_data(config: dict[str, Any]) -> dict[str, Any]:
    universe_scope = config.get("raw_download", {}).get("universe_scope", "core")
    universe_path = resolve_path(config["universes"][universe_scope])
    universe = load_universe(universe_path)

    raw_outputs, summary_frame = fetch_universe_history(config)
    raw_panel = pd.concat(
        [pd.read_csv(path) for path in raw_outputs.values()],
        ignore_index=True,
    )
    calendar = build_trading_calendar(raw_panel)
    calendar_path = save_table(calendar, path_for("raw_reference_root") / "trading_calendar.csv")
    reference_outputs = fetch_reference_data(config, universe)
    index_outputs = fetch_index_history(config)

    manifest = {
        "source": config.get("source", "KBS"),
        "listing_source": config.get("listing_source", "VCI"),
        "industry_listing_source": config.get("industry_listing_source", "KBS"),
        "symbol_source_overrides": {
            str(key).upper(): str(value).upper()
            for key, value in config.get("symbol_source_overrides", {}).items()
        },
        "start_date": str(config["start_date"]),
        "end_date": _effective_end_date(config),
        "interval": config.get("interval", "1D"),
        "universe_scope": universe_scope,
        "symbols": universe,
        "price_adjustment": config.get("raw_download", {}).get("price_adjustment"),
        "value_mode": config.get("raw_download", {}).get("value_mode"),
        "raw_schema": RAW_COLUMNS,
        "rows": int(len(raw_panel)),
        "symbols_fetched": int(raw_panel["symbol"].nunique()),
        "symbols_by_source": {
            str(key): int(value)
            for key, value in raw_panel.groupby("source")["symbol"].nunique().to_dict().items()
        },
    }
    manifest_path = write_json(manifest, path_for("raw_reference_root") / "raw_data_manifest.json")

    return {
        "raw_files": raw_outputs,
        "index_files": index_outputs,
        "reference_files": {**reference_outputs, "trading_calendar": calendar_path, "manifest": manifest_path},
        "summary_rows": int(len(summary_frame)),
    }
