import pytest
from unittest.mock import patch, MagicMock
from src.data_models.GroupSeparator import GroupSeparator
from src.data_models.QuestionDetails import QuestionDetails
from src.data_models.CompletionResponse import CompletionResponse

class TestGroupSeparator:

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
        self.group_separator = GroupSeparator(question_ids=[1, 2])

    def test_initialization(self):
        assert self.group_separator.question_ids == [1, 2]
        assert self.group_separator.grouping_response is None
        assert self.group_separator.grouped_questions is None

    @patch('src.data_models.GroupSeparator.get_gpt_prediction_via_proxy')
    def test_fetch_grouping_response(self, mock_get_gpt_prediction_via_proxy):
        mock_response = MagicMock(spec=CompletionResponse)
        mock_response.content = '{"Group 1": [1], "Group 2": [2]}'
        mock_get_gpt_prediction_via_proxy.return_value = mock_response

        self.group_separator.fetch_grouping_response()

        assert self.group_separator.grouping_response == mock_response
        assert self.group_separator.grouped_questions == {
            "Group 1": [1],
            "Group 2": [2]
        }

    @patch('src.data_models.GroupSeparator.get_gpt_prediction_via_proxy')
    def test_raise_exception_if_can_not_parse_json(self, mock_get_gpt_prediction_via_proxy):
        mock_response = MagicMock(spec=CompletionResponse)
        mock_response.content = 'Invalid JSON'
        mock_get_gpt_prediction_via_proxy.return_value = mock_response

        with pytest.raises(ValueError):
            self.group_separator.fetch_grouping_response()

    def test_make_messages_for_group_separator(self):
        from src.data_models.GroupSeparator import make_messages_for_group_separator
        messages = make_messages_for_group_separator(self.group_separator.question_details_dict)
        
        assert isinstance(messages, list)
        assert len(messages) == 2
        assert 'system' in messages[0]['role']
        assert 'user' in messages[1]['role']

