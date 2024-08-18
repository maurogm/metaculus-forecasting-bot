import pytest
from src.data_models.CompletionResponse import CompletionResponse

class TestCompletionResponse:
    
    response_json_example = {
        'id': 'chatcmpl-something',
        'object': 'chat.completion',
        'created': 12345678,
        'model': 'gpt-4o-2024-05-13',
        'choices': [
            {
                'index': 0,
                'message': {
                    'role': 'assistant',
                    'content': '...Here comes a lot of text...'
                },
                'logprobs': None,
                'finish_reason': 'stop'
            }
        ],
        'usage': {
            'prompt_tokens': 863,
            'completion_tokens': 398,
            'total_tokens': 1261
        },
        'system_fingerprint': 'fp_something'
    }

    def test_properties(self):
        completion_response = CompletionResponse(self.response_json_example)
        
        assert completion_response.id == 'chatcmpl-something'
        assert completion_response.object == 'chat.completion'
        assert completion_response.model == 'gpt-4o-2024-05-13'
        assert completion_response.first_choice == self.response_json_example['choices'][0]
        assert completion_response.content == '...Here comes a lot of text...'
        assert completion_response.finish_reason == 'stop'
        assert completion_response.prompt_tokens == 863
        assert completion_response.completion_tokens == 398
        assert completion_response.total_tokens == 1261
        assert completion_response.tokens_all == '863, 398, 1261'
    
    def test_empty_choices(self):
        response_json = {
            'id': 'chatcmpl-something',
            'object': 'chat.completion',
            'created': 12345678,
            'model': 'gpt-4o-2024-05-13',
            'choices': [],
            'usage': {
                'prompt_tokens': 863,
                'completion_tokens': 398,
                'total_tokens': 1261
            },
            'system_fingerprint': 'fp_something'
        }
        completion_response = CompletionResponse(response_json)
        
        assert completion_response.first_choice is None
        assert completion_response.content is None
        assert completion_response.finish_reason is None
    
    def test_missing_usage(self):
        response_json = {
            'id': 'chatcmpl-something',
            'object': 'chat.completion',
            'created': 12345678,
            'model': 'gpt-4o-2024-05-13',
            'choices': [
                {
                    'index': 0,
                    'message': {
                        'role': 'assistant',
                        'content': '...Here comes a lot of text...'
                    },
                    'logprobs': None,
                    'finish_reason': 'stop'
                }
            ],
            'system_fingerprint': 'fp_something'
        }
        completion_response = CompletionResponse(response_json)
        
        assert completion_response.prompt_tokens is None
        assert completion_response.completion_tokens is None
        assert completion_response.total_tokens is None
        assert completion_response.tokens_all == 'None, None, None'
    
    def test_partial_usage(self):
        response_json = {
            'id': 'chatcmpl-something',
            'object': 'chat.completion',
            'created': 12345678,
            'model': 'gpt-4o-2024-05-13',
            'choices': [
                {
                    'index': 0,
                    'message': {
                        'role': 'assistant',
                        'content': '...Here comes a lot of text...'
                    },
                    'logprobs': None,
                    'finish_reason': 'stop'
                }
            ],
            'usage': {
                'prompt_tokens': 863
            },
            'system_fingerprint': 'fp_something'
        }
        completion_response = CompletionResponse(response_json)
        
        assert completion_response.prompt_tokens == 863
        assert completion_response.completion_tokens is None
        assert completion_response.total_tokens is None
        assert completion_response.tokens_all == '863, None, None'
    
    def test_missing_keys(self):
        """Test that if keys are missing, the properties return None."""
        response_json = {}
        completion_response = CompletionResponse(response_json)
        
        assert completion_response.id is None
        assert completion_response.object is None
        assert completion_response.model is None
        assert completion_response.first_choice is None
        assert completion_response.content is None
        assert completion_response.finish_reason is None
        assert completion_response.prompt_tokens is None
        assert completion_response.completion_tokens is None
        assert completion_response.total_tokens is None
        assert completion_response.tokens_all == 'None, None, None'

