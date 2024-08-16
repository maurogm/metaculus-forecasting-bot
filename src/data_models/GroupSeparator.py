from typing import Dict, Iterable, List

from src.config import OPENAI_MODEL, logger_factory

from src.data_models.CompletionResponse import CompletionResponse
from src.data_models.QuestionDetails import QuestionDetails

from src.metaculus import get_all_question_details_from_ids, extract_questions
from src.openai_utils import get_gpt_prediction_via_proxy
from src.utils import try_to_find_and_eval_dict


class GroupSeparator:
    """
    Class to handle the grouping of questions with a similar scope.

    Parameters
    ----------
    question_ids : Iterable[int]
        List of question IDs to group.

    Attributes
    ----------
    question_ids : Iterable[int]
        List of question IDs to group.
    question_details_dict : Dict[int, QuestionDetails]
        Dictionary with the question details.
    grouping_response : CompletionResponse
        Response from the OpenAI API.
    grouped_questions : Dict[str, List[int]]
        Dictionary with the grouped questions.

    Methods
    -------
    fetch_grouping_response()
        Fetches the grouping response from the OpenAI API.

    Examples
    --------
    >>> group_separator = GroupSeparator([1, 2, 3])
    >>> group_separator.fetch_grouping_response()
    >>> groups_dictionary = group_separator.grouped_questions
    """

    def __init__(self, question_ids: Iterable[int]):
        self.logger = logger_factory.make_logger(name="GroupSeparator")

        if not isinstance(question_ids, Iterable):
            raise ValueError(f"question_ids must be a list of integers, not {
                             type(question_ids)}")

        self.question_ids = question_ids
        self.question_details_dict: Dict[int, QuestionDetails] = get_all_question_details_from_ids(
            self.question_ids)
        self.grouping_response: CompletionResponse = None
        self.grouped_questions: Dict[str, List[int]] = None

    def fetch_grouping_response(self):
        if self.grouping_response is not None:
            self.logger.warning(
                "Tried to fetch grouping response when it was already fetched.")
        else:
            messages = make_messages_for_group_separator(
                self.question_details_dict)
            self.grouping_response = get_gpt_prediction_via_proxy(
                messages, model=OPENAI_MODEL)
            try:
                self.grouped_questions = try_to_find_and_eval_dict(
                    self.grouping_response.content)
            except:
                self.logger.error(f"Failed to parse the following question grouping content:\n```\n{
                             self.grouping_response.content}\n```\n")
                raise ValueError("Failed to parse question grouping content.")


def make_messages_for_group_separator(question_details_dict: Dict[int, QuestionDetails]) -> List[Dict[str, str]]:
    """
    Generates a message for the OpenAI API.
    """

    system_message = """
You will be provided a dictionary with a set of questions. The keys are the question IDs and the values are the question themselves.

Your task is to group the questions that are extremely related to each other.
 
We will consider that this is the case when the questions are very correlated to each other.
It is not enough that they belong to the same topic, such as "Finance" or "Bascketball".
Soy they have to be asking about the exact same subject (but maybe from a different angle), to be in the same group.
Questions that are unrelated should be in their own group.

For each group, you should provide a brief descriptor that summarizes the topic that is being asked about.
Phrase the descriptos as a search query that you would use to find information about the topic on the internet.

Your answer must be a JSON object, where the keys are the group descriptors and the values are the question IDs that belong to that group.
It must not contain any other piece of text, comments, or symbols, so that it can be parsed correctly.
An example of the expected format is:

{
"group_descriptor1": [question_id1, question_id2, ...],
"group_descriptor2": [question_id3, ...],
"group_descriptor3": [question_id4, question_id5, ...],
...
}
"""

    questions_dict = extract_questions(question_details_dict)
    user_message = f"""Here is the dictionary of questions:
```
{questions_dict}
```
"""

    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message}
    ]
    return messages
