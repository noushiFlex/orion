# Data sources

Raw downloads live in `data/raw/<folder>/` and are never modified by code. Every raw file is listed in
`data/manifest.jsonl` (source, URL, SHA-256, collection time; `fetched_at` is null for files that predate the
manifest). Personal use only: football-data.co.uk and Footiqo restrict use to private, non-commercial purposes.

## Used by Orion (registered sources)

### football_data -> data/raw/football-data-co-uk/
- https://football-data.co.uk, `<DIV>_<SEASON>.csv` (e.g. `E0_1617.csv`), 22 leagues x 2015/16 to 2026/27, plus
  `fixtures.csv` (scheduled matches with odds; re-fetched each time).
- Results, shots, shots on target, corners, fouls, cards, referee (not every league), Pinnacle closing odds
  (`PSCH/PSCD/PSCA`). Reference source for team names.

### understat -> data/raw/understat/
- `https://understat.com/getLeagueData/<league>/<start year>`, raw JSON `<DIV>_<SEASON>.json`, Big Five only.
- Per-match goals and xG. Unofficial endpoint, may change. Names mapped automatically (167/167 accepted).

### clubelo -> data/raw/xgabora-club-football/
- https://github.com/xgabora/Club-Football-Match-Data. `EloRatings.csv`: ClubElo snapshots on the 1st and 15th
  of each month since 2000 (the ClubElo API returns 502). Values after 2025-06-01 are the mirror author's
  provisional continuation, not ClubElo's. `Matches.csv` is used only to learn team-name mappings.
- Names mapped automatically (166 accepted, aliases such as "MGladbach"/"M'gladbach" kept).

### transfermarkt -> data/raw/transfermarkt/transfermarkt-datasets.duckdb
- https://github.com/dcaribou/transfermarkt-datasets. Used for the `lineup` feature (v2): starting XIs and player
  valuations, top divisions E0 SC0 D1 I1 SP1 F1 N1 B1 P1 T1 G1. Names mapped automatically (330/331).
- Frozen upstream: games stop 2026-07-06, valuations 2026-06-12. Also holds 1,871 Champions League games.

## Benchmarks (not model inputs)

- `data/raw/theopenmodel/`: https://theopenmodel.com/data/, CC BY 4.0 (credit "The Open Model",
  theopenmodel.com). `predictions.csv`: their probabilities locked before kickoff, 2026/27 Big Five, results
  filled after full time. `fixtures-2026-27.csv`: probabilities for all 1,752 matches of the season. Built on
  ClubElo, so no new information as a feature; used to compare Orion out of sample.

## Downloaded, not used yet

- `data/raw/football-data-co-uk-extra/`: 16 extra leagues (ARG AUT BRA CHN DNK FIN IRL JPN MEX NOR POL ROU RUS SWE
  SWZ USA), one multi-season file each, about 63,800 matches. Different column layout (`Home, Away, HG, AG`).
- `data/raw/statsbomb/`: all 18 Champions League matches in StatsBomb Open Data (events with shot xG, lineups).
  Credit StatsBomb if anything is published.
- `data/raw/salimt-football-datasets/repo/`: 143k player injuries with dates (per player, not per match).
- `data/raw/football-injuries/repo/`: scraper code only; `mappings/team_map.csv` may help later.

## Tested, not available

- ClubElo API: HTTP 502. FBref: HTTP 403, advanced stats gone since January 2026. `soccerdata`: fails to download
  its TLS library here. Footiqo: licence forbids automated downloads of its free files.
- Open-Meteo works, but no stadium coordinates yet, and kickoff times are missing for many matches.

## Gaps

- Pre-match lineups, injuries and suspensions per match.
- Shot-level xG for league matches; Champions League stats beyond results.
- Stadium coordinates for weather.
