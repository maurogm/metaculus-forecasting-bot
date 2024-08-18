import pytest
from unittest.mock import patch, MagicMock
from src.data_models.AskNewsFetcher import AskNewsFetcher

class TestAskNewsFetcher:

    @pytest.fixture(autouse=True)
    def setup(self):
        self.query = "Test Query"
        self.ask_news_fetcher = AskNewsFetcher(query=self.query, n_hot_articles=5, n_historical_articles=10)

    def test_initialization(self):
        assert self.ask_news_fetcher.query == self.query
        assert self.ask_news_fetcher.n_hot_articles == 5
        assert self.ask_news_fetcher.n_historical_articles == 10
        assert self.ask_news_fetcher.hot_response is None
        assert self.ask_news_fetcher.historical_response is None

    @patch('src.data_models.AskNewsFetcher.AskNewsSDK')
    def test_fetch_articles(self, mock_asknews_sdk):
        mock_sdk_instance = mock_asknews_sdk.return_value
        mock_hot_response = MagicMock()
        mock_historical_response = MagicMock()

        mock_sdk_instance.news.search_news.side_effect = [mock_hot_response, mock_historical_response]

        self.ask_news_fetcher.fetch_articles()

        assert self.ask_news_fetcher.hot_response == mock_hot_response
        assert self.ask_news_fetcher.historical_response == mock_historical_response


    @patch('src.data_models.AskNewsFetcher.AskNewsSDK')
    def test_fetch_articles_with_no_articles(self, mock_asknews_sdk):
        mock_sdk_instance = mock_asknews_sdk.return_value
        mock_sdk_instance.news.search_news.return_value = None

        self.ask_news_fetcher.fetch_articles()

        assert self.ask_news_fetcher.hot_response is None
        assert self.ask_news_fetcher.historical_response is None

    @patch('src.data_models.AskNewsFetcher.AskNewsSDK')
    def test_make_news_str(self, mock_asknews_sdk):
        # Mock responses
        mock_sdk_instance = mock_asknews_sdk.return_value
        mock_hot_response = MagicMock()
        mock_hot_response.as_string = "Hot news summary."
        mock_historical_response = MagicMock()
        mock_historical_response.as_string = "Historical news summary."

        mock_sdk_instance.news.search_news.side_effect = [mock_hot_response, mock_historical_response]

        self.ask_news_fetcher.fetch_articles()
        news_str = self.ask_news_fetcher.make_news_str()

        assert "Hot news summary." in news_str
        assert "Historical news summary." in news_str
