import pytest
from datetime import datetime
from src.data_models.QuestionDetails import QuestionDetails



# Tests for the QuestionDetails class:
class TestQuestionDetails:
    details_dict_example = {
            'id': 1,
            'title': 'Test Title',
            'resolution_criteria': 'Test Resolution Criteria',
            'fine_print': 'Test Fine Print',
            'description': 'Test Description',
            'publish_time': '2021-12-13T14:15:16'
        }
    
    def test_properties(self):
        details_dict = {
            'id': 1,
            'title': 'Test Title',
            'resolution_criteria': 'Test Resolution Criteria',
            'fine_print': 'Test Fine Print',
            'description': 'Test Description',
            'publish_time': '2021-10-10T10:10:10'
        }
        details = QuestionDetails(details_dict)
        assert details.id == 1
        assert details.title == 'Test Title'
        assert details.resolution_criteria == 'Test Resolution Criteria'
        assert details.fine_print == 'Test Fine Print'
        assert details.background == 'Test Description'
        assert details.publish_time == datetime(2021, 10, 10, 10, 10, 10)
        assert details.publish_date == '2021-10-10'
    
    def remove_key(self, dictionary, key):
        new_dict = dictionary.copy()
        del new_dict[key]
        return new_dict
    
    def test_fail_if_missing_key(self):
        """The class should not be able to be instantiated if a required key is missing."""
        for key in ['id', 'title', 'resolution_criteria', 'fine_print', 'description', 'publish_time']:
            details_dict = self.remove_key(self.details_dict_example, key)
            with pytest.raises(ValueError):
                QuestionDetails(details_dict)