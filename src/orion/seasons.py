"""Season codes follow football-data.co.uk: '1516' is 2015/16. Compare seasons with start_year, not as strings."""


def code(start: int) -> str:
    return f"{start % 100:02d}{(start + 1) % 100:02d}"


def start_year(season: str) -> int:
    return 2000 + int(season[:2])


def span(first: str, last: str) -> list[str]:
    return [code(y) for y in range(start_year(first), start_year(last) + 1)]
