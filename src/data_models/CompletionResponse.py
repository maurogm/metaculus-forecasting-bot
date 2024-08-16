from dataclasses import dataclass
from typing import Dict, Any, Optional

@dataclass
class CompletionResponse:
    response_json: Dict[str, Any]

    @property
    def id(self) -> Optional[int]:
        return self.response_json.get('id')

    @property
    def object(self) -> Optional[str]:
        return self.response_json.get('object')

    @property
    def model(self) -> Optional[str]:
        return self.response_json.get('model')

    @property
    def first_choice(self) -> Optional[str]:
        maybe_choices = self.response_json.get('choices')
        if maybe_choices is not None:
            return maybe_choices[0]
        else:
            return None

    @property
    def content(self) -> Optional[str]:
        try:
            return self.first_choice.get('message').get('content')
        except:
            return None

    @property
    def finish_reason(self) -> Optional[str]:
        try:
            return self.response_json.get('finish_reason')
        except:
            return None

    @property
    def prompt_tokens(self) -> Optional[int]:
        try:
            return self.response_json.get('usage').get('prompt_tokens')
        except:
            return None

    @property
    def completion_tokens(self) -> Optional[int]:
        try:
            return self.response_json.get('usage').get('completion_tokens')
        except:
            return None

    @property
    def total_tokens(self) -> Optional[int]:
        try:
            return self.response_json.get('usage').get('total_tokens')
        except:
            return None
    
    @property
    def tokens_all(self) -> Optional[int]:
        "A string with prompt_tokens, completion_tokens, total_tokens"
        return f"{self.prompt_tokens}, {self.completion_tokens}, {self.total_tokens}"
