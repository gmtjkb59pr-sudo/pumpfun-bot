import asyncio
import unittest
from unittest.mock import patch

from pumpfun_bot.social_metadata import fetch_has_socials


class _FakeResponse:
    def __init__(self, json_data=None, status=200):
        self._json_data = json_data
        self.status = status

    async def json(self, content_type=None):
        return self._json_data

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False


class _FakeSession:
    def __init__(self, response):
        self._response = response

    def get(self, url, timeout=None):
        return self._response

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False


def _patched(response):
    return patch("pumpfun_bot.social_metadata.aiohttp.ClientSession", return_value=_FakeSession(response))


class FetchHasSocialsTests(unittest.TestCase):
    def test_true_when_twitter_present(self):
        response = _FakeResponse({"name": "Test", "twitter": "https://x.com/test"})
        with _patched(response):
            self.assertTrue(asyncio.run(fetch_has_socials("https://example.invalid/meta.json")))

    def test_true_when_telegram_present(self):
        response = _FakeResponse({"name": "Test", "telegram": "https://t.me/test"})
        with _patched(response):
            self.assertTrue(asyncio.run(fetch_has_socials("https://example.invalid/meta.json")))

    def test_false_when_no_social_fields(self):
        response = _FakeResponse({"name": "Test", "description": "no socials here"})
        with _patched(response):
            self.assertFalse(asyncio.run(fetch_has_socials("https://example.invalid/meta.json")))

    def test_false_when_social_field_is_empty_string(self):
        response = _FakeResponse({"name": "Test", "twitter": ""})
        with _patched(response):
            self.assertFalse(asyncio.run(fetch_has_socials("https://example.invalid/meta.json")))

    def test_false_on_non_200_status(self):
        response = _FakeResponse({"twitter": "https://x.com/test"}, status=404)
        with _patched(response):
            self.assertFalse(asyncio.run(fetch_has_socials("https://example.invalid/meta.json")))

    def test_false_when_metadata_is_not_a_dict(self):
        response = _FakeResponse(["not", "a", "dict"])
        with _patched(response):
            self.assertFalse(asyncio.run(fetch_has_socials("https://example.invalid/meta.json")))

    def test_false_when_uri_is_empty(self):
        self.assertFalse(asyncio.run(fetch_has_socials("")))

    def test_false_on_fetch_exception(self):
        class _RaisingSession:
            def get(self, url, timeout=None):
                raise TimeoutError("simulated timeout")

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                return False

        with patch("pumpfun_bot.social_metadata.aiohttp.ClientSession", return_value=_RaisingSession()):
            self.assertFalse(asyncio.run(fetch_has_socials("https://example.invalid/meta.json")))


if __name__ == "__main__":
    unittest.main()
