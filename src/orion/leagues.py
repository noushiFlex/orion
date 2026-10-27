"""League metadata. Supporting a new league = one line here, then its code in a config."""

from dataclasses import dataclass, replace


@dataclass(frozen=True)
class League:
    code: str  # football-data.co.uk division code
    country: str
    tier: int
    name: str
    understat: str | None = None  # Understat league key, when Understat covers it
    transfermarkt: str | None = None  # Transfermarkt competition id, when the Transfermarkt dataset covers it
    source: str = "football_data"  # registered source that provides its matches


_TM = {"E0": "GB1", "SC0": "SC1", "D1": "L1", "I1": "IT1", "SP1": "ES1", "F1": "FR1", "N1": "NL1", "B1": "BE1",
       "P1": "PO1", "T1": "TR1", "G1": "GR1"}

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
    # football-data "new" leagues: one multi-season file each, results and closing odds only.
    League("ARG", "ARG", 1, "Liga Profesional", source="football_data_extra"),
    League("AUT", "AUT", 1, "Bundesliga (AT)", source="football_data_extra"),
    League("BRA", "BRA", 1, "Serie A (BR)", source="football_data_extra"),
    League("CHN", "CHN", 1, "Super League (CN)", source="football_data_extra"),
    League("DNK", "DNK", 1, "Superliga", source="football_data_extra"),
    League("FIN", "FIN", 1, "Veikkausliiga", source="football_data_extra"),
    League("IRL", "IRL", 1, "Premier Division (IE)", source="football_data_extra"),
    League("JPN", "JPN", 1, "J1 League", source="football_data_extra"),
    League("MEX", "MEX", 1, "Liga MX", source="football_data_extra"),
    League("NOR", "NOR", 1, "Eliteserien", source="football_data_extra"),
    League("POL", "POL", 1, "Ekstraklasa", source="football_data_extra"),
    League("ROU", "ROU", 1, "Liga I", source="football_data_extra"),
    League("RUS", "RUS", 1, "Premier League (RU)", source="football_data_extra"),
    League("SWE", "SWE", 1, "Allsvenskan", source="football_data_extra"),
    League("SWZ", "SWZ", 1, "Super League (CH)", source="football_data_extra"),
    League("USA", "USA", 1, "MLS", source="football_data_extra"),
]
LEAGUES = {league.code: replace(league, transfermarkt=_TM.get(league.code)) for league in _ALL}


def get(code: str) -> League:
    try:
        return LEAGUES[code]
    except KeyError:
        raise KeyError(f"unknown league {code!r}; add it to orion/leagues.py") from None
