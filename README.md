# Pump.fun Trading Bot (Solana)

Een lokale bot voor pump.fun met drie combineerbare strategieën:
- **Sniper** — koopt nieuwe tokens vlak na launch die door je filters komen
- **Copy trading** — spiegelt trades van wallets die je aanwijst
- **Market maker / grid** — plaatst grid buy/sell orders rond de laatste prijs

Plus een alerting-systeem (console + optioneel Telegram) en een centrale
risk-manager die harde limieten afdwingt over alle strategieën heen.

## ⚠️ Lees dit eerst

- **pump.fun tokens zijn extreem speculatief.** De overgrote meerderheid gaat
  naar (bijna) nul. Rugpulls, wash trading en bot-vs-bot snipe-wars zijn de norm,
  niet de uitzondering.
- **Dit is geen officiële pump.fun software.** De bot gebruikt de community
  API van [PumpPortal](https://pumpportal.fun/) omdat pump.fun zelf geen
  publieke trading-API aanbiedt. Endpoints/velden kunnen zonder aankondiging
  wijzigen — test altijd eerst met `dry_run: true`.
- **Er zit geen garantie op winst in.** Filters in de sniper-strategie
  (liquiditeit, social links) zijn basale ruis-filters, geen scam-detectie.
- **Zet nooit meer op je hot wallet dan je kunt missen.** Gebruik een aparte
  wallet, niet je hoofd-wallet.
- Ik heb dit gebouwd als een werkend startpunt met code-review in gedachten —
  niet als kant-en-klaar productiesysteem. Test grondig, begin klein, en lees
  de code voordat je hem live zet.

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
PUMPPORTAL_API_KEY=          # alleen nodig als je later de Lightning API gebruikt
TELEGRAM_BOT_TOKEN=          # optioneel, voor Telegram alerts
```

**Nooit** je `.env` of `config.yaml` (met echte wallet-verwijzingen) delen of
in git committen. Gebruik idealiter een losse, klein-gevulde wallet.

Pas daarna `config.yaml` aan:
1. Zet een fatsoenlijke RPC url (public mainnet-beta RPC is traag/rate-limited;
   overweeg Helius, QuickNode of Triton voor een snipers/copy-trading use case)
2. Zet welke strategie(ën) je `enabled: true` wilt
3. Controleer de `risk:` sectie — dit zijn je enige bescherming tegen een bug
   die te veel/te vaak handelt

## Starten

```bash
python main.py
```

Standaard staat `risk.dry_run: true`: de bot logt en stuurt alerts over wat
hij zou doen, zonder echte transacties te versturen. Kijk het gedrag een tijd
aan voordat je `dry_run: false` zet.

## Dashboard (visueel volgen in de browser)

Naast de terminal-logs heeft de bot een lokaal webdashboard. Zodra je
`python main.py` start, zie je in de log een regel als:

```
Dashboard: open http://localhost:8765 in je browser om de bot te volgen.
```

Open die URL in je browser terwijl de bot draait. Je ziet:
- Een grote status-banner: rustig blauw bij **DRY RUN**, pulserend rood bij
  **LIVE** — zo zie je in één oogopslag of er echt geld op het spel staat.
- Open exposure, dag P&L, trades/uur, uptime — allemaal tegen je limieten uit
  `config.yaml`.
- Een live feed van alle alerts (snipe-kandidaten, aankopen, fouten).
- Een tabel met recente trades (dry-run én echt).

Het dashboard gebruikt geen extra dependencies (alleen Python's ingebouwde
webserver) en luistert standaard alleen op `127.0.0.1` — dus niet bereikbaar
vanaf andere apparaten in je netwerk. Wil je het uitzetten of op een andere
poort draaien, pas dan de `dashboard:` sectie in `config.yaml` aan.

## Data-logging & "learning stats"

Elke trade en alert wordt (naast het in-memory dashboard) ook append-only
weggeschreven naar `data/activity_log.jsonl` (gitignored), zodat je een volledige
geschiedenis hebt die een dashboard-restart overleeft.

De sniper-strategie probeert daarnaast bij te houden wat er na een simulated
buy met de koers gebeurde (checkpoints op 60s/300s/900s), zodat je op het
dashboard onder "Learning stats" kunt zien of bv. tokens met socials of meer
liquiditeit het beter deden. **Belangrijk:** dit vereist PumpPortal's
`subscribeTokenTrade` feed, wat zelf weer een **gefunde API key** vereist
(0.02+ SOL op de wallet achter `PUMPPORTAL_API_KEY`). Zonder zo'n key wijst
PumpPortal de subscription af en toont het dashboard duidelijk "N snipe(s)
konden niet gevolgd worden" in plaats van foutieve 0%-cijfers te verzinnen.
Dit raakt ook de market-maker strategie, die dezelfde feed gebruikt.

Dit bouwt alleen de dataverzameling — er is geen automatische aanpassing van
filters/strategie op basis van deze stats. Dat "echt leren" zou een aparte,
grotere stap zijn.

## Structuur

```
main.py                          - start alle enabled strategieën + dashboard
pumpfun_bot/config.py            - laadt en valideert config.yaml + .env
pumpfun_bot/risk.py              - centrale risicolimieten (ALLE orders gaan hierdoorheen)
pumpfun_bot/pumpportal_client.py - WebSocket data feed + order-uitvoering
pumpfun_bot/alerts.py            - console/Telegram meldingen
pumpfun_bot/state.py             - gedeelde status voor het dashboard
pumpfun_bot/dashboard_server.py  - lokale webserver (http://localhost:8765)
pumpfun_bot/static/dashboard.html - de dashboard-pagina zelf
pumpfun_bot/strategies/sniper.py
pumpfun_bot/strategies/copytrade.py
pumpfun_bot/strategies/market_maker.py
```

## Bekende beperkingen / TODO

- Sniper voert nu automatisch take-profit/stop-loss/timeout uit (zie
  `pumpfun_bot/outcome_tracker.py`), zowel dry-run als live. In LIVE modus
  wordt een echte sell verstuurd (100% van de holding) via PumpPortal's
  Local Trading API; een positie wordt alleen als gesloten beschouwd als die
  sell daadwerkelijk lukt - bij een mislukte sell blijft de positie open en
  wordt het opnieuw geprobeerd, zodat het dashboard nooit "verkocht" toont
  terwijl de wallet de tokens nog vasthoudt. Er is geen partial-exit (altijd
  100%) en geen trailing stop.
- Market maker gebruikt een vereenvoudigd prijsmodel; houdt geen rekening met
  de exacte bonding-curve wiskunde van pump.fun.
- Market maker (en outcome-tracking, zie hierboven) hangen af van PumpPortal's
  `subscribeTokenTrade`/`subscribeAccountTrade` feeds, die alleen werken met
  een **gefunde API key** (0.02+ SOL). Zonder key wijst PumpPortal de
  subscription af en doet market maker dus niets (geen grid wordt ooit
  geïnitialiseerd) — dit is niet eerder expliciet getest/gedocumenteerd.
- Geen persistente opslag van posities/PnL tussen herstarts (alles zit in het
  geheugen, dus een restart reset de risk-manager teller).
- PumpPortal veldnamen (bv. `vSolInBondingCurve`, `txType`) zijn gebaseerd op
  hun publieke voorbeelden; verifieer dit tegen een paar live WebSocket
  berichten voordat je live handelt, want APIs veranderen.

## Testen

```bash
python -m unittest discover -s tests -v
```

Dekt vooralsnog de risk-manager (`pumpfun_bot/risk.py`), omdat alle orders
daar doorheen moeten en het de belangrijkste veiligheidslaag is.

## Support / feedback

Als iets niet werkt zoals verwacht: lees eerst de logs (ze zijn expres
uitgebreid), en check of de PumpPortal API structuur nog matcht met wat er in
`pumpportal_client.py` en de strategieën staat verwacht.
