import pytest
from unittest.mock import patch, MagicMock
from src.data_models.DetailsPreparation import DetailsPreparation
from src.data_models.QuestionDetails import QuestionDetails
from src.data_models.CompletionResponse import CompletionResponse

class TestDetailsPreparation:

    @pytest.fixture(autouse=True)
    def setup(self):
        self.mock_question_details = {
            1: QuestionDetails({
                'id': 1,
                'title': 'Question 1',
                'resolution_criteria': 'Criteria 1',
                'fine_print': 'Fine print 1',
                'description': 'Description 1',
                'publish_time': '2023-08-18T00:00:00'
            }),
            2: QuestionDetails({
                'id': 2,
                'title': 'Question 2',
                'resolution_criteria': 'Criteria 2',
                'fine_print': 'Fine print 2',
                'description': 'Description 2',
                'publish_time': '2023-08-19T00:00:00'
            })
        }
        self.details_preparation = DetailsPreparation(
            question_ids=[1, 2], 
            question_details_dict=self.mock_question_details
        )

    def test_initialization_with_one_question(self):
        details_prep = DetailsPreparation(
            question_ids=[1], 
            question_details_dict=self.mock_question_details
        )
        assert details_prep.unified_details == self.mock_question_details[1].details_dict

    def test_initialization_with_multiple_questions(self):
        assert self.details_preparation.unified_details is None

    @patch('src.data_models.DetailsPreparation.get_gpt_prediction_via_proxy')
    def test_fetch_detail_unification_response(self, mock_get_gpt_prediction_via_proxy):
        """Test that the Fetch method fills the unified_details attribute with a mocked response from the API."""
        mock_response = MagicMock(spec=CompletionResponse)
        mock_response.content = '{"title": "Unified Title", "background": "Unified Background", "resolution_criteria": "Unified Criteria", "fine_print": "Unified Fine Print"}'
        mock_get_gpt_prediction_via_proxy.return_value = mock_response

        self.details_preparation.fetch_detail_unification_response()

        assert self.details_preparation.unified_details['title'] == 'Unified Title'
        assert self.details_preparation.unified_details['background'] == 'Unified Background'
        assert self.details_preparation.unified_details['resolution_criteria'] == 'Unified Criteria'
        assert self.details_preparation.unified_details['fine_print'] == 'Unified Fine Print'

    def test_make_details_str_without_fetching(self):
        with pytest.raises(ValueError):
            self.details_preparation.make_details_str()

    def test_make_details_str_with_fetched_details(self):
        self.details_preparation.unified_details = {
            "title": "Unified Title",
            "background": "Unified Background",
            "resolution_criteria": "Unified Criteria",
            "fine_print": "Unified Fine Print"
        }

        details_str = self.details_preparation.make_details_str()
        #assert "Unified Title" in details_str #The title is not currently included in the prompt
        assert "Unified Background" in details_str
        assert "Unified Criteria" in details_str
        assert "Unified Fine Print" in details_str
