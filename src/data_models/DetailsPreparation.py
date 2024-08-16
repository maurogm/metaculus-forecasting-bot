import json

from typing import Dict, Iterable, List

from src.config import OPENAI_MODEL, logger_factory

from src.data_models.CompletionResponse import CompletionResponse
from src.data_models.QuestionDetails import QuestionDetails

from src.metaculus import get_all_question_details_from_ids, extract_questions
from src.openai_utils import get_gpt_prediction_via_proxy
from src.utils import try_to_find_and_eval_dict


class DetailsPreparation:
    """
    Class to handle the unification of the details of similar questions.

    If there is only one question, there is no call to the LLM, and the details are simply extracted from the question_details_dict.
    If there are two or more questions, the details are unified into a single string by calling the LLM.

    Parameters
    ----------
    question_ids : Iterable[int]
        List of question IDs to group.
    question_details_dict : Dict[int, QuestionDetails], optional
        Dictionary with the question details.

    Attributes
    ----------
    question_ids : Iterable[int]
        List of question IDs to group.
    question_details_dict : Dict[int, QuestionDetails]
        Dictionary with the question details. Might be provided as an argument.
    unification_response : CompletionResponse
        Response from the OpenAI API.
    unified_details : Dict[str, str]
        Dictionary with the unified details parsed from the unification_response.
        Mandatory fields are: "title", "background", "resolution_criteria", "fine_print".

    Methods
    -------
    fetch_detail_unification_response()
        Fetches the response from the OpenAI API.
    make_details_str()
        Generates a unified string with the details of the questions to be forecasted.

    Examples
    --------
    >>> details_unificator = DetailsPreparaton([1, 2, 3])
    >>> details_unificator.fetch_detail_unification_response()
    >>> unified_details_str = details_unificator.make_details_str()
    """

    def __init__(self, question_ids: Iterable[int], question_details_dict: Dict[int, QuestionDetails] = None):
        self.logger = logger_factory.make_logger(name="DetailsUnificator")

        if not isinstance(question_ids, Iterable):
            raise ValueError(f"question_ids must be a list of integers, not {
                             type(question_ids)}")

        self.question_ids = question_ids
        if question_details_dict is None:
            self.question_details_dict: Dict[int, QuestionDetails] = get_all_question_details_from_ids(
                self.question_ids)
        else:
            self.question_details_dict = question_details_dict
        self.unification_response: CompletionResponse = None

        # If there is only one question, there is no need to unify:
        if len(self.question_ids) == 1:
            self.unified_details = self.question_details_dict[self.question_ids[0]].details_dict
        else:
            self.unified_details: Dict[str, str] = None

    def fetch_detail_unification_response(self):
        """
        Fetches the response from the OpenAI API.
        """
        self.logger.debug(f"Fetching detail unification response for question IDs: {self.question_ids}")
        if self.unified_details is None:
            messages = make_messages_for_details_unification(
                self.question_details_dict, self.question_ids)
            self.unification_response = get_gpt_prediction_via_proxy(
                messages, model=OPENAI_MODEL)
            try:
                unified_details_dict = try_to_find_and_eval_dict(
                    self.unification_response.content)
                self.unified_details = unified_details_dict
            except:
                self.logger.error(f"Failed to parse the following detail unification content:\n```\n{
                    self.unification_response.content}\n```\n")
                raise ValueError("Failed to parse detail unification content.")

    @property
    def concatenated_questions_str(self):
        question_list = extract_questions(self.question_details_dict)
        formated_questions = [
            f"- question_id={q_id}: {question_list[q_id]}" for q_id in self.question_ids]
        concatenated_questions = "\n".join(formated_questions)
        return f"Following are the questions that must be answered, preceded by their respective question IDs:\n{concatenated_questions}"

    def make_details_str(self):
        """
        Generates a unified string with the details of the questions to be forecasted.
        """
        if self.unified_details is None:
            self.logger.error(
                "Tried to call DetailsUnificator.question_details_str without fetching the unified details first.")
            raise ValueError(
                "Unified details have not been fetched yet, so unified_details is None.")
        return apply_question_template_to_unification_json(self.concatenated_questions_str, self.unified_details)


def make_messages_for_details_unification(question_details_dict: Dict[int, QuestionDetails], question_ids: Iterable[int]) -> List[Dict[str, str]]:
    """
    Generates a prompt for unifying the details of similar questions.

    Parameters:
    - question_details_dict (Dict[int, QuestionDetails]): A dictionary where keys are question IDs and values are dictionaries containing question details.
    - question_ids (Iterable[int]): An iterable of question IDs to be unified. It is assumed that all the questions in the iterable are indeed similar.

    Returns:
    - str: A formatted string containing the prompt for unifying the details of the specified questions.
    """

    questions_details = [question_details_dict[q_id] for q_id in question_ids]
    question_str_list = [make_question_str(
        details) for details in questions_details]
    user_message = "\n".join(question_str_list)

    system_message = """
You will recieve a series of similar questions. Each question has it's own background information and resolution criteria, even though they might be very similar.

Your task is to synthesize the information from all the questions in a group and provide a unified background and resolution criteria for the group.
You will do this by deleting information that is repeated in all the questions, and keeping information that is different between the questions.
Your task is only to eliminate redundancies, no information should be lost in this process.

It is okay for the unified question to now be a series of different questions, though grouped together.

Provide your answer in the following JSON format:

{{
    "title": "string",
    "background": "string",
    "resolution_criteria": "string",
    "fine_print": "string"
}}
"""

    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message}
    ]
    return messages


def make_question_str(question_details: QuestionDetails) -> str:
    """
    Given the question details, generates a string with the relevant information for the grouping task.
    """

    QUESTION_TEMPLATE = """
The following are the details of the question with ID {id}:

Title: "{title}"

The Resolution Criteria for the question is:
```
{resolution_criteria}
```

The resolution has the following fine print:
```
{fine_print}
```

Some background information was provided (at {publish_time}), to give context to the question:
```
{background}
```

This is the end of the details of question {id}.
"""

    return QUESTION_TEMPLATE.format(
        id=question_details.id,
        title=question_details.title,
        news_articles="no news_articles here",
        # today=today,
        publish_time=question_details.publish_date,
        background=question_details.background,
        resolution_criteria=question_details.resolution_criteria,
        fine_print=question_details.fine_print,
    )


def collapse_questions_into_str(question_ids: Iterable[int], question_details_dict: Dict[int, QuestionDetails]) -> str:
    """
    Generates a string that clearly states the original questions to be answered, and their IDs.

    Intended to be embedded in the final prompt.
    """
    assert len(
        question_ids) > 0, "question_ids must contain at least 1 question ID"
    question_list = extract_questions(question_details_dict)
    collapsed = "\n".join(
        [f"- question_id={q_id}: {question_list[q_id]}" for q_id in question_ids])
    return f"""
Following are the questions that must be answered, preceded by their respective question IDs:
{collapsed}
    """


def apply_question_template_to_unification_json(question_str: str,
                                                details_dict: Dict) -> str:
    """
    Generates a unified string with the details of the questions to be forecasted.
    """

    resolution = details_dict["resolution_criteria"]
    fine_print = details_dict["fine_print"]
    try:
        background = details_dict["background"]
    except:
        background = details_dict["description"]

    return f"""{question_str}

The Resolution Criteria is:
```
{resolution}
```

The resolution has the following fine print:
```
{fine_print}
```

The following background information was provided to give some context to the question:
```
{background}
```
"""
