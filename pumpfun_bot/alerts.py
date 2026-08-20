"""Stuurt meldingen naar console en optioneel Telegram."""
from __future__ import annotations

import logging

import aiohttp

from .state import bot_state

logger = logging.getLogger("pumpfun_bot.alerts")


class Alerter:
    def __init__(self, console: bool, telegram_enabled: bool, bot_token: str, chat_id: str):
        self.console = console
        self.telegram_enabled = telegram_enabled and bool(bot_token) and bool(chat_id)
        self.bot_token = bot_token
        self.chat_id = chat_id

    async def send(self, message: str) -> None:
        bot_state.log_alert(message)

        if self.console:
            logger.info("ALERT | %s", message)

        if self.telegram_enabled:
            url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
            payload = {"chat_id": self.chat_id, "text": message}
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(url, json=payload, timeout=10) as resp:
                        if resp.status != 200:
                            body = await resp.text()
                            logger.warning("Telegram alert mislukt (%s): %s", resp.status, body)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Telegram alert exception: %s", exc)
