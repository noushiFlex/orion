"""League metadata. Supporting a new league = one line here, then its code in a config."""

from dataclasses import dataclass


@dataclass(frozen=True)
class League:
    code: str  # football-data.co.uk division code
    country: str
    tier: int
    name: str
    understat: str | None = None  # Understat league key, when Understat covers it


_ALL = [
    League("E0", "ENG", 1, "Premier League", "EPL"),
    League("E1", "ENG", 2, "Championship"),
    League("E2", "ENG", 3, "League One"),
    League("E3", "ENG", 4, "League Two"),
    League("EC", "ENG", 5, "National League"),
    League("SC0", "SCO", 1, "Premiership"),
    League("SC1", "SCO", 2, "Championship"),
    League("SC2", "SCO", 3, "League One"),
    League("SC3", "SCO", 4, "League Two"),
    League("D1", "GER", 1, "Bundesliga", "Bundesliga"),
    League("D2", "GER", 2, "2. Bundesliga"),
    League("I1", "ITA", 1, "Serie A", "Serie_A"),
    League("I2", "ITA", 2, "Serie B"),
    League("SP1", "ESP", 1, "La Liga", "La_liga"),
    League("SP2", "ESP", 2, "Segunda Division"),
    League("F1", "FRA", 1, "Ligue 1", "Ligue_1"),
    League("F2", "FRA", 2, "Ligue 2"),
    League("N1", "NED", 1, "Eredivisie"),
    League("B1", "BEL", 1, "Pro League"),
    League("P1", "POR", 1, "Primeira Liga"),
    League("T1", "TUR", 1, "Super Lig"),
    League("G1", "GRE", 1, "Super League"),
]
LEAGUES = {league.code: league for league in _ALL}


def get(code: str) -> League:
    try:
        return LEAGUES[code]
    except KeyError:
        raise KeyError(f"unknown league {code!r}; add it to orion/leagues.py") from None
