import asyncio
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from pumpfun_bot.scam_social_check import (
    evaluate_social_links,
    load_known_scam_links,
    record_scam_links,
    twitter_link_is_live,
    twitter_link_looks_real,
    website_looks_real,
)


class _FakeTextResponse:
    def __init__(self, body="", status=200):
        self._body = body
        self.status = status

    async def text(self, errors="ignore"):
        return self._body

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False


class _FakeSession:
    def __init__(self, response):
        self._response = response

    def get(self, url, timeout=None, allow_redirects=True):
        return self._response

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False


def _patched(response):
    return patch("pumpfun_bot.scam_social_check.aiohttp.ClientSession", return_value=_FakeSession(response))


class TwitterLinkLooksRealTests(unittest.TestCase):
    def test_real_x_com_handle_passes(self):
        self.assertTrue(twitter_link_looks_real("https://x.com/realproject"))

    def test_real_twitter_com_handle_passes(self):
        self.assertTrue(twitter_link_looks_real("https://twitter.com/realproject"))

    def test_handle_with_query_string_still_passes(self):
        self.assertTrue(twitter_link_looks_real("https://x.com/realproject?ref=pumpfun"))

    def test_a_link_to_an_unrelated_domain_fails(self):
        self.assertFalse(twitter_link_looks_real("https://example.invalid/realproject"))

    def test_a_bare_domain_with_no_handle_fails(self):
        self.assertFalse(twitter_link_looks_real("https://x.com/"))

    def test_a_handle_over_fifteen_chars_fails(self):
        self.assertFalse(twitter_link_looks_real("https://x.com/thisusernameiswaytoolongtobereal"))

    def test_empty_string_fails(self):
        self.assertFalse(twitter_link_looks_real(""))


class TwitterLinkIsLiveTests(unittest.TestCase):
    def test_a_real_profile_with_real_content_passes(self):
        response = _FakeTextResponse('<title>Real Project (@realproject) / X</title>')
        with _patched(response):
            self.assertTrue(asyncio.run(twitter_link_is_live("https://x.com/realproject")))

    def test_a_404_status_fails(self):
        response = _FakeTextResponse("User Profile Not Found - X | 404 Error", status=404)
        with _patched(response):
            self.assertFalse(asyncio.run(twitter_link_is_live("https://x.com/neverregistered")))

    def test_a_200_status_with_a_not_found_marker_fails(self):
        # confirmed live 2026-08-24: X can serve the "doesn't exist" page
        # itself at 200 in some cases - don't trust status code alone
        response = _FakeTextResponse("This account doesn't exist", status=200)
        with _patched(response):
            self.assertFalse(asyncio.run(twitter_link_is_live("https://x.com/gone")))

    def test_a_suspended_account_fails(self):
        response = _FakeTextResponse("Account suspended", status=200)
        with _patched(response):
            self.assertFalse(asyncio.run(twitter_link_is_live("https://x.com/suspended")))

    def test_a_connection_failure_fails(self):
        class _RaisingSession:
            def get(self, url, timeout=None, allow_redirects=True):
                raise TimeoutError("simulated timeout")

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                return False

        with patch("pumpfun_bot.scam_social_check.aiohttp.ClientSession", return_value=_RaisingSession()):
            self.assertFalse(asyncio.run(twitter_link_is_live("https://x.com/unreachable")))


class WebsiteLooksRealTests(unittest.TestCase):
    def test_a_real_page_with_real_content_passes(self):
        response = _FakeTextResponse("<html><body>" + "A real project site. " * 20 + "</body></html>")
        with _patched(response):
            self.assertTrue(asyncio.run(website_looks_real("https://real-project.invalid")))

    def test_non_200_status_fails(self):
        response = _FakeTextResponse("whatever", status=404)
        with _patched(response):
            self.assertFalse(asyncio.run(website_looks_real("https://dead-link.invalid")))

    def test_a_near_empty_body_fails(self):
        response = _FakeTextResponse("<html></html>")
        with _patched(response):
            self.assertFalse(asyncio.run(website_looks_real("https://blank.invalid")))

    def test_a_parked_domain_page_fails(self):
        body = "<html><body>" + "This domain may be for sale. Contact us! " * 10 + "</body></html>"
        response = _FakeTextResponse(body)
        with _patched(response):
            self.assertFalse(asyncio.run(website_looks_real("https://parked.invalid")))

    def test_a_connection_failure_fails(self):
        class _RaisingSession:
            def get(self, url, timeout=None, allow_redirects=True):
                raise TimeoutError("simulated timeout")

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                return False

        with patch("pumpfun_bot.scam_social_check.aiohttp.ClientSession", return_value=_RaisingSession()):
            self.assertFalse(asyncio.run(website_looks_real("https://unreachable.invalid")))


class KnownScamLinksStoreTests(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self._path = Path(self._tmpdir.name) / "known_scam_social_links.json"
        self._patcher = patch("pumpfun_bot.scam_social_check.KNOWN_SCAM_LINKS_PATH", self._path)
        self._patcher.start()

    def tearDown(self):
        self._patcher.stop()
        self._tmpdir.cleanup()

    def test_empty_set_when_file_missing(self):
        self.assertEqual(load_known_scam_links(), set())

    def test_recorded_links_are_loadable(self):
        record_scam_links(["https://x.com/scammer1", "https://fake-site.invalid"])
        self.assertEqual(
            load_known_scam_links(), {"https://x.com/scammer1", "https://fake-site.invalid"},
        )

    def test_recording_dedupes_against_existing_links(self):
        record_scam_links(["https://x.com/scammer1"])
        record_scam_links(["https://x.com/scammer1", "https://x.com/scammer2"])
        self.assertEqual(
            load_known_scam_links(), {"https://x.com/scammer1", "https://x.com/scammer2"},
        )

    def test_recording_an_empty_list_is_a_no_op(self):
        record_scam_links([])
        self.assertFalse(self._path.exists())


class EvaluateSocialLinksTests(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self._path = Path(self._tmpdir.name) / "known_scam_social_links.json"
        self._patcher = patch("pumpfun_bot.scam_social_check.KNOWN_SCAM_LINKS_PATH", self._path)
        self._patcher.start()

    def tearDown(self):
        self._patcher.stop()
        self._tmpdir.cleanup()

    def test_no_links_at_all_is_not_sus(self):
        is_sus, reason = asyncio.run(evaluate_social_links({}))
        self.assertFalse(is_sus)
        self.assertIsNone(reason)

    def test_a_fake_twitter_link_is_sus_without_needing_a_website_check(self):
        is_sus, reason = asyncio.run(evaluate_social_links({"twitter": "https://example.invalid/scam"}))
        self.assertTrue(is_sus)
        self.assertIn("twitter", reason)

    def test_a_real_looking_and_live_twitter_with_no_website_is_not_sus(self):
        response = _FakeTextResponse('<title>Real Project (@realproject) / X</title>')
        with _patched(response):
            is_sus, reason = asyncio.run(evaluate_social_links({"twitter": "https://x.com/realproject"}))
        self.assertFalse(is_sus)
        self.assertIsNone(reason)

    def test_a_real_looking_but_dead_twitter_handle_is_sus(self):
        response = _FakeTextResponse("User Profile Not Found - X | 404 Error", status=404)
        with _patched(response):
            is_sus, reason = asyncio.run(evaluate_social_links({"twitter": "https://x.com/neverregistered"}))
        self.assertTrue(is_sus)
        self.assertIn("twitter", reason)

    def test_a_dead_website_is_sus(self):
        response = _FakeTextResponse("", status=404)
        with _patched(response):
            is_sus, reason = asyncio.run(evaluate_social_links({"website": "https://dead.invalid"}))
        self.assertTrue(is_sus)
        self.assertIn("website", reason)

    def test_a_reused_known_scam_link_is_flagged_immediately_no_network_call(self):
        record_scam_links(["https://x.com/repeatoffender"])

        class _ExplodingSession:
            def get(self, *a, **kw):
                raise AssertionError("should not make a network call for a known-scam link")

        with patch("pumpfun_bot.scam_social_check.aiohttp.ClientSession", return_value=_ExplodingSession()):
            is_sus, reason = asyncio.run(
                evaluate_social_links({"twitter": "https://x.com/repeatoffender"})
            )
        self.assertTrue(is_sus)
        self.assertIn("hergebruikt", reason)


if __name__ == "__main__":
    unittest.main()
