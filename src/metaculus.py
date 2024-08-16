from typing import Iterable, Dict, List, Optional
import requests
import json
from src.config import AUTH_HEADERS, API_BASE_URL
from src.data_models.QuestionDetails import QuestionDetails
from src.config import logger_factory

logger = logger_factory.make_logger(name=__name__)


def post_question_comment(question_id, comment_text):
    response = requests.post(
        f"{API_BASE_URL}/comments/",
        json={
            "comment_text": comment_text,
            "submit_type": "N",
            "include_latest_prediction": True,
            "question": question_id,
        },
        **AUTH_HEADERS,
    )
    response.raise_for_status()


def post_question_prediction(question_id, prediction_probability):
    """
    Posts a prediction to a question.

    Prediction probability should be a float between 0 and 1 representing the probability of the event happening.
    """
    url = f"{API_BASE_URL}/questions/{question_id}/predict/"
    response = requests.post(
        url,
        json={"prediction": float(prediction_probability)},
        **AUTH_HEADERS,
    )
    response.raise_for_status()


def upload_predictions(question_ids: List[int], probabilities_dict: Dict[int, float], summaries_dict: Optional[Dict[int, str]]):
    """
    Uploads predictions to Metaculus for a list of questions.

    Optionally, it can also post comments for each question.

    Args:
        question_ids (List[int]): List of question ids.
        forecasts_dict (Dict[int, float]): Dictionary of predictions made for each question id.
        summaries_dict (Optional[Dict[int, str]]): Dictionary of question ids to summaries. If absent, no comments are posted.
    """
    forecasts = [probabilities_dict[question_id] for question_id in question_ids]
    assert all (0 <= forecast <= 1 for forecast in forecasts), "Forecasts should be between 0 and 1"
    for question_id in question_ids:
        prediction_percent = probabilities_dict[question_id]
        logger.info(f"Posting prediction for question {question_id}: {prediction_percent}")
        post_question_prediction(question_id, prediction_percent)
        if summaries_dict is not None:
            prediction_comment = summaries_dict[question_id]
            logger.info(f"Posting comment for question {question_id}: {prediction_comment}")
            post_question_comment(question_id, prediction_comment)

def get_question_details(question_id: int) -> QuestionDetails:
    """
    Gets the details of a question given its id.

    This function makes a GET request to the Metaculus API to get the details of a question given its id.
    """
    url = f"{API_BASE_URL}/questions/{question_id}/"
    response = requests.get(
        url,
        **AUTH_HEADERS,
    )
    response.raise_for_status()
    response_dict = json.loads(response.content)
    return QuestionDetails(response_dict)


def list_questions(tournament_id, offset=0, count=100):
    url_qparams = {
        "limit": count,
        "offset": offset,
        "has_group": "false",
        "order_by": "-activity",
        "forecast_type": "binary",
        "project": tournament_id,
        "status": "open",
        "type": "forecast",
        "include_description": "true",
    }
    url = f"{API_BASE_URL}/questions/"
    response = requests.get(url, **AUTH_HEADERS, params=url_qparams)
    response.raise_for_status()
    data = json.loads(response.content)
    return data


def get_all_question_details_from_ids(question_ids: Iterable[int]) -> Dict[int, QuestionDetails]:
    """
    Given a list of question ids, return a dictionary of question details
    """
    return {q_id: get_question_details(q_id) for q_id in question_ids}


def extract_ids_from_question_list(question_list: Iterable, drop_predicted = False) -> List[int]:
    """
    Given a list of questions, return a list of question ids.

    If drop_predicted is True, only return the ids of questions that have not been predicted yet.
    """
    if drop_predicted:
        return [q['id'] for q in question_list['results'] if q['my_predictions'] is None]
    else:
        return [q['id'] for q in question_list['results']]


def extract_questions(question_details_dict: Dict[int, QuestionDetails]) -> Dict[int, str]:
    """
    Given a dictionary of question details, return a dictionary of question id to question title
    """
    questions_dict = {id: details.title
                      for id, details in question_details_dict.items()}
    return questions_dict

def drop_answered_questions(question_list):
    """
    Given a list of questions, filter out the ones that have been answered
    """
    # TODO: Implement filter. For now, return the same list.

    return question_list