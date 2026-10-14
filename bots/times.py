"""Times the user types (spec section 2): the PC's local time by default,
because that is what the Binance app shows; UTC with --utc."""
from datetime import datetime, timezone, tzinfo

FORMATS = ("%Y-%m-%d %H:%M", "%Y-%m-%dT%H:%M")


def parse_time(text: str, utc: bool, local_tz: tzinfo | None = None) -> datetime:
    for fmt in FORMATS:
        try:
            naive = datetime.strptime(text.strip(), fmt)
            break
        except ValueError:
            continue
    else:
        raise ValueError(f"time {text!r} is not in the form YYYY-MM-DD HH:MM")
    if not 2017 <= naive.year <= 2100:     # Binance opened in 2017
        raise ValueError(f"time {text!r} is out of range (years 2017 to 2100)")
    if utc:
        return naive.replace(tzinfo=timezone.utc)
    # astimezone() on a naive time applies the PC zone's rules for that date
    local = naive.replace(tzinfo=local_tz) if local_tz else naive.astimezone()
    return local.astimezone(timezone.utc)
