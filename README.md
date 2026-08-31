# Pump.fun Trading Bot (Solana)

Een lokale, config-gedreven bot voor pump.fun. Strategieën zijn combineerbaar
(alles uit, tot je ze aanzet):

- **Sniper** — koopt nieuwe tokens vlak na launch die door je filters komen
- **Social watch** — wacht op socials in de metadata i.p.v. direct te snipe'en
- **Birdeye movers** — bestaande tokens met een volume/prijs-spike (Birdeye API)
- **CoinGecko movers** — zelfde niche, via CoinGecko Demo API (hoger call-budget)
- **Moonshot hunter** — kleine, bewuste loterij-allocatie op zeldzame 100-1000x
- **Copy trading** — spiegelt trades van wallets die je aanwijst
- **Market maker / grid** — grid buy/sell op tokens in `target_tokens`, met
  echte pump.fun bonding-curve prijzen (constant-product, virtual reserves)

Plus: centrale RiskManager, OutcomeTracker (TP/SL/trailing/ladder), dashboard
op localhost:8765, optioneel Telegram, optioneel Jito (size-gated).

## ⚠️ Lees dit eerst

- **pump.fun tokens zijn extreem speculatief.** De overgrote meerderheid gaat
  naar (bijna) nul. Rugpulls, wash trading en bot-vs-bot snipe-wars zijn de norm,
  niet de uitzondering.
- **Dit is geen officiële pump.fun software.** De bot gebruikt de community
  API van [PumpPortal](https://pumpportal.fun/) omdat pump.fun zelf geen
  publieke trading-API aanbiedt. Endpoints/velden kunnen zonder aankondiging
  wijzigen — test altijd eerst met `dry_run: true`.
- **Er zit geen garantie op winst in.** Filters zijn basale ruis-filters, geen
  scam-detectie. Moonshot hunter heeft bewust geen bewezen edge.
- **Zet nooit meer op je hot wallet dan je kunt missen.** Gebruik een aparte
  wallet, niet je hoofd-wallet.
- Dit is een werkend lokaal startpunt, geen kant-en-klaar productiesysteem.

## Installatie

```bash
cd pumpfun-bot
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt

cp config.example.yaml config.yaml
cp .env.example .env
```

Vul in `.env`:
```
SOLANA_PRIVATE_KEY=<je base58 private key>
PUMPPORTAL_API_KEY=          # verplicht voor subscribeTokenTrade / outcome-tracking
TELEGRAM_BOT_TOKEN=          # optioneel
BIRDEYE_API_KEY=             # alleen birdeye_movers
COINGECKO_API_KEY=           # alleen coingecko_movers
```

**Nooit** je `.env` of `config.yaml` (met echte wallet-verwijzingen) delen of
in git committen.

Pas daarna `config.yaml` aan:
1. Zet een fatsoenlijke RPC url (public mainnet-beta is traag/rate-limited)
2. Zet welke strategie(ën) je `enabled: true` wilt
3. Controleer de `risk:` sectie — dit zijn je enige bescherming tegen een bug
   die te veel/te vaak handelt. `config.example.yaml` zet
   `max_exposure_pct_of_balance: 45` en `max_open_positions: 10`; een bestaande
   `config.yaml` zonder die velden blijft het oude gedrag (0 / 1000).

## Starten

```bash
python main.py
```

Standaard staat `risk.dry_run: true`: de bot logt en stuurt alerts over wat
hij zou doen, zonder echte transacties te signen of te versturen. De
PumpPortal-client weigert in dry_run ook hard om te signen, zodat een
vergeten check in een strategie nooit live kan gaan. Kijk het gedrag een tijd
aan voordat je `dry_run: false` zet.

## Dashboard

Zodra je `python main.py` start:

```
Dashboard: open http://localhost:8765 in je browser om de bot te volgen.
```

Je ziet DRY RUN vs LIVE, open exposure, dag P&L (modeled én echte wallet),
trades/uur, alerts en recente trades. Alleen op `127.0.0.1`.

## Persistence (overleeft een restart)

Alles onder `data/` is gitignored.

- **Open posities** — `pumpfun_bot/position_store.py`, aparte files voor
  dry-run en live (anders probeert een live sessie dry-run-spookposities te
  verkopen). Wallet-reconciliatie gooit holdings die de wallet niet meer
  heeft eruit (na 2 misses), en probeert untracked wallet-tokens te
  liquideren.
- **Risk-tellers** — dagelijks verlies, trades/uur-timestamps, en exposure.
  Exposure wordt bij start **herberekend uit de posities** (inclusief
  `remaining_fraction` na een ladder-rung), zodat het klopt met de wallet.
  Zonder dit reset een restart de dag-limiet en de exposure-cap.

## Exits (OutcomeTracker)

Sniper / social_watch / birdeye / coingecko / moonshot sturen hun buys naar
één OutcomeTracker. Per positie:

- **Take-profit / stop-loss** — vaste drempels, of een `take_profit_ladder`
  (verkoop `sell_pct`% van wat er NOG over is bij elke multiplier).
- **Trailing stop** — armed vanaf `trailing_activation_pct`, exit als prijs
  `trailing_stop_pct` van de piek terugvalt. Werkt ook naast een ladder
  (beschermt wat er nog vastzit, ook als de eerste rung nog niet is geraakt).
- **Timeout / stale price** — force-exit als er te lang geen tick is, of na
  `max_hold_sec` (moonshot heeft hier eigen, veel langere waarden).
- **Live sells** gaan via PumpPortal Local Trading API. Een mislukte sell
  laat de positie open (nooit "verkocht" op het dashboard terwijl de wallet
  de tokens nog heeft). Percentage-sell (`amount: "100%"`) is de bewezen
  path; amount-based sell (exacte UI-decimal hoeveelheid) is opt-in
  (`use_absolute_amount_sell`) en valt altijd terug op percentage als die
  faalt. PumpPortal's index heeft ~15s nodig na een buy voordat een
  percentage-sell werkt (`MIN_SELL_DELAY_SEC`).

## Jito (optioneel, size-gated)

`use_jito_bundles_for_take_profit` / `use_jito_bundles_for_sniper_buys`
staan default uit. De tip die live nodig was om een bundle te laten landen
is **0.01 SOL**. Bij de echte trade-groottes van deze bot (~0.012–0.053 SOL)
is dat 19–83% van de trade — oneconomisch. `max_jito_tip_pct_of_trade`
(default 10) houdt Jito vanzelf uit tot trades groot genoeg zijn.

## PumpPortal API key (gefund)

`subscribeNewToken` is gratis. `subscribeTokenTrade` en
`subscribeAccountTrade` vereisen een **gefunde API key** (min. 0.02 SOL op
de wallet achter `PUMPPORTAL_API_KEY`) en zijn gemetered (0.01 SOL per
10.000 events). Zonder key:

- Outcome-tracking krijgt geen echte ticks (dashboard toont dat, i.p.v.
  valse 0%-cijfers)
- Market maker initialiseert nooit een grid
- Copytrade ziet geen wallet-trades

Gebruik **één** websocket-verbinding; de client reconnect met exponential
backoff. Ontbrekende verplichte velden (`vSolInBondingCurve`, curve-reserves,
`solAmount`) zijn een skip + warning, niet stiekem 0.

## Data-logging & learning stats

Trades/alerts gaan append-only naar `data/activity_log.jsonl`. De sniper
houdt koerscheckpoints bij (60/300/900s) voor het dashboard. Er is een
optionele auto-tuner (alleen strenger, nooit losser) en schaduw-modellen;
die passen filters niet automatisch aan zonder bewijs.

## Structuur

```
main.py                          - start enabled strategieën + dashboard
pumpfun_bot/config.py            - laadt config.yaml + .env
pumpfun_bot/risk.py              - limieten; tellers persistent
pumpfun_bot/risk_store.py        - data/risk_state_{dry_run|live}.json
pumpfun_bot/position_store.py    - open posities op disk
pumpfun_bot/outcome_tracker.py   - TP/SL/trailing/ladder + live sells
pumpfun_bot/bonding_curve.py     - constant-product virtual reserves
pumpfun_bot/pumpportal_client.py - WS feed + local trading API
pumpfun_bot/dashboard_server.py  - http://localhost:8765
pumpfun_bot/strategies/...
```

## Bekende beperkingen

- PumpPortal veldnamen (`vSolInBondingCurve`, `txType`, …) komen uit hun
  publieke docs; verifieer tegen live WS berichten voor je live handelt.
- Amount-based sells zijn server-side gevalideerd (200 + unsigned tx), niet
  gegarandeerd on-chain; percentage-sell blijft de fallback.
- Market maker is een simpele grid op tokens die jij al target — geen
  inventory-optimalisatie, geen post-graduation AMM-routing voorbij
  `pool: auto`.
- Modeled P&L (fee-model) kan groen zijn terwijl de echte wallet daalt;
  het dashboard toont daarom ook de echte balans t.o.v. sessiestart.
- Copytrade onthoudt "wat we deze sessie zelf kochten" alleen in memory.
- `force_simulated` op sniper/social_watch simuleert die strategie terwijl
  de echte tracker open live-posities blijft beschermen.

## Testen

```bash
python -m unittest discover -s tests -v
```

Geen live RPC of wallet nodig. Dekt o.a. risk-manager + persistence,
bonding-curve market maker, exits (ladder/trailing/amount-fallback),
PumpPortal client (dry_run hard-stop, backoff).

## Support / feedback

Als iets niet werkt: lees eerst de logs (ze zijn expres uitgebreid) en check
of de PumpPortal API nog matcht met `pumpportal_client.py`.
