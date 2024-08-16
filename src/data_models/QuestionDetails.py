from dataclasses import dataclass
from typing import Dict, Optional, Any
from datetime import datetime

@dataclass
class QuestionDetails:
    details_dict: Dict[str, Any]

    def __post_init__(self):
        required_keys = ['id', 'title', 'resolution_criteria', 'fine_print', 'description', 'publish_time']
        missing_keys = [key for key in required_keys if key not in self.details_dict]
        if missing_keys:
            raise ValueError(f"Missing required keys in details_dict: {missing_keys}")
    
    @property
    def id(self) -> Optional[int]:
        return self.details_dict.get('id')
    
    @property
    def title(self) -> Optional[str]:
        return self.details_dict.get('title')
    
    @property
    def resolution_criteria(self) -> Optional[str]:
        return self.details_dict.get('resolution_criteria')
    
    @property
    def fine_print(self) -> Optional[str]:
        return self.details_dict.get('fine_print', "")
    
    @property
    def background(self) -> Optional[str]:
        return self.details_dict.get('description')
    
    @property
    def publish_time(self) -> Optional[datetime]:
        maybe_time = self.details_dict.get('publish_time')
        if maybe_time is None:
            return None
        else:
            return datetime.fromisoformat(maybe_time)

    @property
    def publish_date(self) -> Optional[str]:
        if self.publish_time is None:
            return None
        else:
            return self.publish_time.strftime("%Y-%m-%d")