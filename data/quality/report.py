from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class Report:
    symbol: str
    timeframe: str
    checked_from: datetime | None
    checked_to: datetime | None
    total_candles: int
    duplicates: int
    invalid: int
    missing: int
    completeness_pct: float
    verdict: str
    details: dict = field(default_factory=dict)

    def render(self) -> str:
        d = self.details or {}
        explained = d.get("explained_missing", 0)
        missing = f"{self.missing:,}"
        if explained:
            missing += f"  (+{explained:,} in explained exchange outages)"
        out = (
            f"\n{self.symbol}  [{self.timeframe}]\n"
            f"\nCandles:\n{self.total_candles:,}\n"
            f"\nDuplicates:\n{self.duplicates}\n"
            f"\nInvalid:\n{self.invalid}\n"
            f"\nMissing:\n{missing}\n"
            f"\nCompleteness:\n{self.completeness_pct:.6f}%\n"
        )
        if "reason" in d:
            out += f"\nReason:\n{d['reason']}\n"
        if d.get("gap_count"):
            unexplained = d.get("unexplained_gap_count", 0)
            out += (f"\nGaps:\n{unexplained} unexplained, "
                    f"{d['gap_count'] - unexplained} explained\n")
        if d.get("largest_gaps"):
            out += "\nLargest gaps (UTC, [start, end)):\n"
            for g in d["largest_gaps"]:
                start = datetime.fromisoformat(g["start"])
                end = datetime.fromisoformat(g["end"])
                label = "explained outage" if g["explained"] else "UNEXPLAINED"
                out += (f"  {start:%Y-%m-%d %H:%M} -> {end:%Y-%m-%d %H:%M}"
                        f"  {g['minutes']:,} min  {label}\n")
        out += f"\nSTATUS:\n{self.verdict}\n"
        return out
