import pytest
from unittest.mock import patch, MagicMock
from src.data_models.Forecaster import Forecaster
from src.data_models.DetailsPreparation import DetailsPreparation
from src.data_models.CompletionResponse import CompletionResponse
from src.data_models.AskNewsFetcher import AskNewsFetcher

class TestForecaster:

    @pytest.fixture(autouse=True)
    def setup(self):
        # Mock DetailsPreparation
        self.mock_details_preparation = MagicMock(spec=DetailsPreparation)
        self.mock_details_preparation.make_details_str.return_value = "Details about the question"
        self.mock_details_preparation.question_ids = [1, 2]

        # Mock AskNewsFetcher (optional)
        self.mock_news_fetcher = MagicMock(spec=AskNewsFetcher)
        self.mock_news_fetcher.make_news_str.return_value = "News related to the question"

        # Create Forecaster instance
        self.forecaster = Forecaster(details_preparator=self.mock_details_preparation, news=self.mock_news_fetcher)

    def test_initialization(self):
        assert self.forecaster.details_preparator == self.mock_details_preparation
        assert self.forecaster.news == self.mock_news_fetcher
        assert self.forecaster.forecast_response is None
        assert self.forecaster.forecast_dict is None

    @patch('src.data_models.Forecaster.get_gpt_prediction_via_proxy')
    def test_fetch_forecast_response(self, mock_get_gpt_prediction_via_proxy):
        mock_response = MagicMock(spec=CompletionResponse)
        mock_response.content = '{"forecasts": {"1": 0.75}, "summaries": {"1": "Predicted with high confidence"}}'
        mock_get_gpt_prediction_via_proxy.return_value = mock_response

        # Fetch the forecast response
        self.forecaster.fetch_forecast_response()
        self.forecaster.parse_forecast_response()

        # Now check the forecast_dict
        assert self.forecaster.forecast_response == mock_response
        assert self.forecaster.forecast_dict == {
            "forecasts": {1: 0.75},
            "summaries": {1: "Predicted with high confidence"}
        }

    @patch('src.data_models.Forecaster.get_gpt_prediction_via_proxy')
    def test_fetch_forecast_response_with_parsing_error(self, mock_get_gpt_prediction_via_proxy):
        mock_response = MagicMock(spec=CompletionResponse)
        mock_response.content = 'Invalid JSON'
        mock_get_gpt_prediction_via_proxy.return_value = mock_response

        with pytest.raises(ValueError):
            self.forecaster.fetch_forecast_response()
            self.forecaster.parse_forecast_response()

    def test_make_messages_for_forecast_with_news(self):
        messages = self.forecaster.make_messages_for_forecast()
        
        assert isinstance(messages, list)
        assert len(messages) >= 2
        assert 'system' in messages[0]['role']
        assert 'assistant' in messages[1]['role']
        assert 'user' in messages[-1]['role']
        assert "Details about the question" in messages[-1]['content']
        assert "News related to the question" in messages[1]['content']

    def test_make_messages_for_forecast_without_news(self):
        self.forecaster.news = None
        messages = self.forecaster.make_messages_for_forecast()
        
        assert isinstance(messages, list)
        assert len(messages) == 2  # Without news, there should be only system and user messages
        assert 'system' in messages[0]['role']
        assert 'user' in messages[1]['role']
        assert "Details about the question" in messages[1]['content']
        assert "News related to the question" not in messages[-1]['content']
