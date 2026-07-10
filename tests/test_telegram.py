from unittest.mock import MagicMock, patch

from oraclebot.utils.telegram import send_message


def test_send_message_skips_when_not_configured():
    with patch('oraclebot.utils.telegram.requests.post') as mock_post:
        send_message(None, None, "hello")
        mock_post.assert_not_called()


def test_send_message_posts_to_correct_url_with_escaped_text():
    with patch('oraclebot.utils.telegram.requests.post') as mock_post:
        mock_post.return_value = MagicMock(raise_for_status=lambda: None)
        send_message('TOKEN123', 'CHAT456', "Trend: LONG (0.5)")

        mock_post.assert_called_once()
        args, kwargs = mock_post.call_args
        assert args[0] == "https://api.telegram.org/botTOKEN123/sendMessage"
        assert kwargs['data']['chat_id'] == 'CHAT456'
        # MarkdownV2-Sonderzeichen muessen escaped sein (siehe escape_chars in send_message).
        assert r'\(' in kwargs['data']['text'] and r'\)' in kwargs['data']['text']
        assert kwargs['data']['parse_mode'] == 'MarkdownV2'
