"""
Cheap, no-API-key heuristics for judging whether a token's declared social
links (website/twitter, from its pump.fun metadata) look fake, run right
after a real buy so a scam-looking token can be sold immediately instead of
waiting for its price to move.

Deliberately NOT doing real X/Twitter API lookups (account age, follower
count) - user-requested: that needs a paid API tier and adds real per-check
cost and latency for every single buy. What's checked instead is free:
- website: does the URL actually resolve to real content, or is it dead /
  a parked-domain placeholder page (a very common rug-launch tell - the
  "website" field gets filled with a domain nobody ever set up)
- twitter/x link: is it even shaped like a real handle URL, or garbage
  (many rug launches reuse a template that points "twitter" at something
  that isn't twitter.com/x.com at all, or has no handle)
- reused-scam-link: the exact same website/twitter link showing up on a
  later token this bot flagged before - the same scam kit/creator reusing
  identical fake links across multiple launches, confirmed as a real
  pattern worth checking for cheaply once the first store exists
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path

import aiohttp

logger = logging.getLogger("pumpfun_bot.scam_social_check")

KNOWN_SCAM_LINKS_PATH = Path("data/known_scam_social_links.json")

# real X/Twitter handles: 1-15 chars, letters/digits/underscore only -
# anything else in the handle position is not a real profile URL
_TWITTER_HANDLE_RE = re.compile(
    r"^https?://(www\.)?(twitter\.com|x\.com)/([A-Za-z0-9_]{1,15})/?(\?.*)?$"
)

# confirmed live on parked/placeholder domains - a real project's site
# won't contain these
_PARKED_DOMAIN_MARKERS = (
    "domain is for sale",
    "buy this domain",
    "this domain may be for sale",
    "godaddy.com/domainsearch",
    "future home of something quite cool",
    "parking page",
    "domain parking",
)


def load_known_scam_links() -> set[str]:
    if not KNOWN_SCAM_LINKS_PATH.exists():
        return set()
    try:
        data = json.loads(KNOWN_SCAM_LINKS_PATH.read_text())
        return set(data) if isinstance(data, list) else set()
    except Exception:  # noqa: BLE001
        logger.debug("Kon known_scam_social_links.json niet lezen, start leeg.")
        return set()


def record_scam_links(links: list[str]) -> None:
    """Adds newly-flagged links to the persisted store, dedupe-on-write."""
    if not links:
        return
    existing = load_known_scam_links()
    existing.update(links)
    KNOWN_SCAM_LINKS_PATH.parent.mkdir(parents=True, exist_ok=True)
    KNOWN_SCAM_LINKS_PATH.write_text(json.dumps(sorted(existing), indent=2))


def twitter_link_looks_real(url: str) -> bool:
    """Format check only, no network call (see module docstring for why) -
    real X handle URL shape, not just "some string was present"."""
    return bool(_TWITTER_HANDLE_RE.match(url.strip()))


async def website_looks_real(url: str, timeout_sec: float = 5.0) -> bool:
    """A dead link, a non-200 response, a suspiciously empty page, or a
    parked-domain placeholder all count as "not real" - a legitimate
    project's site returns real content. Any network failure is treated as
    NOT real (unlike fetch_has_socials's fail-open default) since this
    check specifically exists to catch exactly that kind of dead/fake link,
    not to be lenient about it."""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=timeout_sec, allow_redirects=True) as resp:
                if resp.status != 200:
                    return False
                body = await resp.text(errors="ignore")
    except Exception as exc:  # noqa: BLE001
        logger.debug("Website %s niet bereikbaar of geen geldige respons: %s", url, exc)
        return False

    stripped = body.strip()
    if len(stripped) < 200:
        return False
    lowered = stripped.lower()
    if any(marker in lowered for marker in _PARKED_DOMAIN_MARKERS):
        return False
    return True


async def evaluate_social_links(links: dict[str, str]) -> tuple[bool, str | None]:
    """Returns (is_sus, reason). Checks the reused-scam-link store first
    (free, no network) before the network-dependent checks."""
    website = links.get("website", "").strip()
    twitter = links.get("twitter", "").strip()

    known_scam = load_known_scam_links()
    reused = [link for link in (website, twitter) if link and link in known_scam]
    if reused:
        return True, f"link hergebruikt van eerder gevlagde scam: {reused[0]}"

    if twitter and not twitter_link_looks_real(twitter):
        return True, f"twitter-link ziet er nep uit (geen echte handle-URL): {twitter}"

    if website and not await website_looks_real(website):
        return True, f"website niet bereikbaar of lijkt een lege/geparkeerde pagina: {website}"

    return False, None
