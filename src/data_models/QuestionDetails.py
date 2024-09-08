from dataclasses import dataclass
from typing import Dict, Optional, Any
from datetime import datetime


@dataclass
class QuestionDetails:
    """
    Dataclass that encapsulates the details of a question.

    This class provides properties to access different parts of the question details.

    Parameters
    ----------
    details_dict : Dict[str, Any]
        Dictionary containing the details of the question.
    """

    details_dict: Dict[str, Any]

    def __post_init__(self):
        required_keys = ['id', 'title', 'resolution_criteria',
                         'fine_print', 'description', 'publish_time']
        missing_keys = [
            key for key in required_keys if key not in self.details_dict]
        if missing_keys:
            raise ValueError(
                f"Missing required keys in details_dict: {missing_keys}")

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
    def active_state(self) -> Optional[str]:
        return self.details_dict.get('active_state')

    @property
    def community_quartiles(self) -> Optional[bool]:
        community_prediction = self.details_dict.get(
            "community_prediction", {}).get("full", {})
        if "q1" in community_prediction:
            return {
                "quartile1": community_prediction.get("q1"),
                "median": community_prediction.get("q2"),
                "quartile3": community_prediction.get("q3"),
            }
        else:
            return None

    @property
    def n_forecasters(self) -> Optional[int]:
        return self.details_dict.get('number_of_forecasters')

    @property
    def resolution(self) -> Optional[bool]:
        return self.details_dict.get('resolution')

    @property
    def publish_time(self) -> Optional[datetime]:
        maybe_time = self.details_dict.get('publish_time')
        return datetime.fromisoformat(maybe_time) if maybe_time else None

    @property
    def publish_date(self) -> Optional[str]:
        return self.publish_time.strftime("%Y-%m-%d") if self.publish_time else None

    @property
    def created_time(self) -> Optional[datetime]:
        maybe_time = self.details_dict.get('created_time')
        return datetime.fromisoformat(maybe_time) if maybe_time else None

    @property
    def created_date(self) -> Optional[str]:
        return self.created_time.strftime("%Y-%m-%d") if self.created_time else None

    @property
    def close_time(self) -> Optional[datetime]:
        maybe_time = self.details_dict.get('close_time')
        return datetime.fromisoformat(maybe_time) if maybe_time else None

    @property
    def close_date(self) -> Optional[str]:
        return self.close_time.strftime("%Y-%m-%d") if self.close_time else None

    @property
    def resolve_time(self) -> Optional[datetime]:
        maybe_time = self.details_dict.get('resolve_time')
        return datetime.fromisoformat(maybe_time) if maybe_time else None

    @property
    def resolve_date(self) -> Optional[str]:
        return self.resolve_time.strftime("%Y-%m-%d") if self.resolve_time else None

    @property
    def last_activity_time(self) -> Optional[datetime]:
        maybe_time = self.details_dict.get('last_activity_time')
        return datetime.fromisoformat(maybe_time) if maybe_time else None

    @property
    def last_activity_date(self) -> Optional[str]:
        return self.last_activity_time.strftime("%Y-%m-%d") if self.last_activity_time else None

    @property
    def activity(self) -> Optional[float]:
        return self.details_dict.get('activity')

    @property
    def comment_count(self) -> Optional[int]:
        return self.details_dict.get('comment_count')

    @property
    def forecast_type(self) -> Optional[str]:
        return self.details_dict.get("possibilities", {}).get("type")

    @property
    def projects(self) -> Optional[str]:
        return self.details_dict.get("projects")

    @property
    def project_ids(self) -> Optional[str]:
        return [project.get("id") for project in self.projects] if self.projects else None
