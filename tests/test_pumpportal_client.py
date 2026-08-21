import unittest

from pumpfun_bot.pumpportal_client import authenticated_ws_url


class AuthenticatedWsUrlTests(unittest.TestCase):
    def test_no_api_key_returns_url_unchanged(self):
        self.assertEqual(
            authenticated_ws_url("wss://pumpportal.fun/api/data", ""),
            "wss://pumpportal.fun/api/data",
        )

    def test_appends_api_key_as_query_param(self):
        self.assertEqual(
            authenticated_ws_url("wss://pumpportal.fun/api/data", "abc123"),
            "wss://pumpportal.fun/api/data?api-key=abc123",
        )

    def test_appends_with_ampersand_if_url_already_has_query_params(self):
        self.assertEqual(
            authenticated_ws_url("wss://pumpportal.fun/api/data?foo=bar", "abc123"),
            "wss://pumpportal.fun/api/data?foo=bar&api-key=abc123",
        )


if __name__ == "__main__":
    unittest.main()
