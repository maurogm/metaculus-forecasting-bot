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
