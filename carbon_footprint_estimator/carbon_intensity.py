"""Fetch six contiguous national GB forecast intervals, using UTC boundaries."""
import json
import math
import ssl
import certifi
from datetime import datetime, timedelta, timezone
from urllib.request import Request, urlopen
from urllib.error import URLError
from .config import API_URL, TIMEOUT_SECONDS


class ForecastError(RuntimeError):
    """Forecasts are missing or cannot be trusted."""


def timestamp(value):
    """Parse an explicitly zoned API timestamp and normalize to UTC."""
    dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        raise ValueError("Missing timestamp timezone.")
    return dt.astimezone(timezone.utc)


def period_start(now):
    """Floor an aware instant to the containing half-hour in UTC."""
    if now.tzinfo is None:
        raise ValueError("An aware datetime is required.")
    dt = now.astimezone(timezone.utc)
    return dt.replace(minute=30*(dt.minute//30), second=0, microsecond=0)


def parse_forecast(payload, now):
    """Require six distinct, adjacent periods with finite forecast values."""
    try:
        start = period_start(now)
        rows = {}
        for row in payload["data"]:
            begin, end = timestamp(row["from"]), timestamp(row["to"])
            if not start <= begin < start+timedelta(hours=3):
                continue
            value = row["intensity"]["forecast"]
            if begin in rows or end-begin != timedelta(minutes=30):
                raise ValueError("Duplicate or invalid forecast period.")
            if isinstance(value, bool) or not isinstance(value, (int,float)) or not math.isfinite(value) or value < 0:
                raise ValueError("Missing or invalid forecast intensity.")
            rows[begin] = dict(start=begin.isoformat(), end=end.isoformat(), forecast=value, unit="gCO2/kWh")
        wanted = [start+timedelta(minutes=30*i) for i in range(6)]
        if set(rows) != set(wanted):
            raise ValueError("Six consecutive forecast periods are required.")
        return [rows[t] for t in wanted]
    except (KeyError, TypeError, ValueError, AttributeError) as exc:
        raise ForecastError("Invalid carbon-intensity forecast: " + str(exc)) from exc


def fetch_forecast(now=None):
    """Fetch forecasts with a timeout; never fall back to actual or invented values."""
    now = now or datetime.now(timezone.utc)
    start = period_start(now)
    finish = start+timedelta(hours=3)
    fmt = lambda d: d.strftime("%Y-%m-%dT%H:%MZ")
    request = Request(API_URL+"/"+fmt(start)+"/"+fmt(finish), headers={"Accept":"application/json", "User-Agent":"PromptCarbon-CFE/1.0"})
    try:
        with urlopen(request, timeout=TIMEOUT_SECONDS, context=ssl.create_default_context(cafile=certifi.where())) as response:
            payload = json.load(response)
        return parse_forecast(payload, now)
    except (URLError, OSError, ValueError) as exc:
        raise ForecastError("Carbon-intensity forecasts unavailable: " + str(exc)) from exc
