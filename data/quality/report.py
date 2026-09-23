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
        return (
            f"\n{self.symbol}  [{self.timeframe}]\n"
            f"\nCandles:\n{self.total_candles:,}\n"
            f"\nDuplicates:\n{self.duplicates}\n"
            f"\nInvalid:\n{self.invalid}\n"
            f"\nMissing:\n{self.missing}\n"
            f"\nCompleteness:\n{self.completeness_pct:.6f}%\n"
            f"\nSTATUS:\n{self.verdict}\n"
        )
