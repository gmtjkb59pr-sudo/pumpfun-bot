import asyncio
import unittest
from unittest.mock import patch

from pumpfun_bot.coingecko import fetch_trending_pools, parse_pool_candidate

SOL_MINT = "So11111111111111111111111111111111111111112"
USDC_MINT = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"


def _pool(
    base_id=f"solana_TARGETpump", quote_id=f"solana_{SOL_MINT}",
    price_change=None, market_cap_usd="500000.0", volume_h24="123456.0", name="TARGET / SOL",
    pool_created_at=None,
):
    return {
        "attributes": {
            "name": name,
            "market_cap_usd": market_cap_usd,
            "price_change_percentage": price_change or {
                "m5": "1.2", "m15": "2.3", "m30": "3.4", "h1": "4.5", "h6": "5.6", "h24": "6.7",
            },
            "volume_usd": {"h24": volume_h24},
            "pool_created_at": pool_created_at,
        },
        "relationships": {
            "base_token": {"data": {"id": base_id}},
            "quote_token": {"data": {"id": quote_id}},
        },
    }


class ParsePoolCandidateTests(unittest.TestCase):
    def test_extracts_the_base_token_as_mint_when_quote_is_sol(self):
        result = parse_pool_candidate(_pool())
        self.assertEqual(result["mint"], "TARGETpump")

    def test_extracts_the_quote_token_as_mint_when_base_is_sol(self):
        pool = _pool(base_id=f"solana_{SOL_MINT}", quote_id="solana_TARGETpump")
        result = parse_pool_candidate(pool)
        self.assertEqual(result["mint"], "TARGETpump")

    def test_extracts_the_base_token_as_mint_when_quote_is_usdc(self):
        pool = _pool(quote_id=f"solana_{USDC_MINT}")
        result = parse_pool_candidate(pool)
        self.assertEqual(result["mint"], "TARGETpump")

    def test_returns_none_when_neither_side_is_a_known_quote_currency(self):
        pool = _pool(base_id="solana_AAAA", quote_id="solana_BBBB")
        self.assertIsNone(parse_pool_candidate(pool))

    def test_returns_none_when_both_sides_are_known_quote_currencies(self):
        pool = _pool(base_id=f"solana_{SOL_MINT}", quote_id=f"solana_{USDC_MINT}")
        self.assertIsNone(parse_pool_candidate(pool))

    def test_returns_none_when_relationships_are_missing(self):
        pool = {"attributes": {}, "relationships": {}}
        self.assertIsNone(parse_pool_candidate(pool))

    def test_returns_none_on_malformed_pool(self):
        self.assertIsNone(parse_pool_candidate({}))
        self.assertIsNone(parse_pool_candidate({"attributes": {}}))

    def test_parses_all_price_change_windows(self):
        result = parse_pool_candidate(_pool(price_change={
            "m5": "10.0", "m15": "20.0", "m30": "30.0", "h1": "40.0", "h6": "50.0", "h24": "60.0",
        }))
        self.assertEqual(result["price_change_pct"], {
            "m5": 10.0, "m15": 20.0, "m30": 30.0, "h1": 40.0, "h6": 50.0, "h24": 60.0,
        })

    def test_missing_price_change_window_is_none_not_zero(self):
        result = parse_pool_candidate(_pool(price_change={"m5": "10.0"}))
        self.assertEqual(result["price_change_pct"]["m5"], 10.0)
        self.assertIsNone(result["price_change_pct"]["h24"])

    def test_missing_market_cap_is_none(self):
        result = parse_pool_candidate(_pool(market_cap_usd=None))
        self.assertIsNone(result["market_cap_usd"])

    def test_market_cap_is_parsed_as_float(self):
        result = parse_pool_candidate(_pool(market_cap_usd="1234567.89"))
        self.assertAlmostEqual(result["market_cap_usd"], 1234567.89)

    def test_missing_volume_is_none(self):
        result = parse_pool_candidate(_pool(volume_h24=None))
        self.assertIsNone(result["volume_24h_usd"])

    def test_pair_name_is_carried_through(self):
        result = parse_pool_candidate(_pool(name="COOL / SOL"))
        self.assertEqual(result["pair_name"], "COOL / SOL")

    def test_pool_created_at_is_parsed_to_a_unix_timestamp(self):
        # user-requested 2026-08-24 ("how can i test more to be sure this
        # strategy will work") - confirmed live: real pool_created_at
        # values range from minutes to 841 DAYS old, previously fetched
        # but never parsed - this is what lets a real candidate be tagged
        # "revival" vs "fresh"
        result = parse_pool_candidate(_pool(pool_created_at="2026-08-23T11:44:49Z"))
        self.assertIsNotNone(result["pool_created_ts"])
        # sanity: a real, plausible unix timestamp, not zero/garbage
        self.assertGreater(result["pool_created_ts"], 1_700_000_000)

    def test_missing_pool_created_at_is_none_not_an_error(self):
        result = parse_pool_candidate(_pool(pool_created_at=None))
        self.assertIsNone(result["pool_created_ts"])

    def test_malformed_pool_created_at_is_none_not_an_error(self):
        result = parse_pool_candidate(_pool(pool_created_at="not-a-real-timestamp"))
        self.assertIsNone(result["pool_created_ts"])


class FetchTrendingPoolsTests(unittest.TestCase):
    def test_returns_none_on_non_200_status(self):
        class FakeResp:
            status = 401
            async def json(self):
                return {}
            async def __aenter__(self):
                return self
            async def __aexit__(self, *a):
                return False

        class FakeSession:
            def get(self, *a, **k):
                return FakeResp()
            async def __aenter__(self):
                return self
            async def __aexit__(self, *a):
                return False

        with patch("pumpfun_bot.coingecko.aiohttp.ClientSession", return_value=FakeSession()):
            result = asyncio.run(fetch_trending_pools("fake-key"))
        self.assertIsNone(result)

    def test_returns_none_on_network_error(self):
        class FailingSession:
            def get(self, *a, **k):
                raise ConnectionError("boom")
            async def __aenter__(self):
                return self
            async def __aexit__(self, *a):
                return False

        with patch("pumpfun_bot.coingecko.aiohttp.ClientSession", return_value=FailingSession()):
            result = asyncio.run(fetch_trending_pools("fake-key"))
        self.assertIsNone(result)

    def test_returns_the_data_list_on_success(self):
        class FakeResp:
            status = 200
            async def json(self):
                return {"data": [{"id": "pool1"}, {"id": "pool2"}]}
            async def __aenter__(self):
                return self
            async def __aexit__(self, *a):
                return False

        class FakeSession:
            def get(self, *a, **k):
                return FakeResp()
            async def __aenter__(self):
                return self
            async def __aexit__(self, *a):
                return False

        with patch("pumpfun_bot.coingecko.aiohttp.ClientSession", return_value=FakeSession()):
            result = asyncio.run(fetch_trending_pools("fake-key"))
        self.assertEqual(result, [{"id": "pool1"}, {"id": "pool2"}])


if __name__ == "__main__":
    unittest.main()
