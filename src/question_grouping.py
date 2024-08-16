from src.data_models.DetailsPreparation import collapse_questions_into_str
from src.metaculus import extract_questions
from typing import Dict, Iterable
from src.data_models.QuestionDetails import QuestionDetails


def apply_template_for_question_grouping(question_details_dict: Dict[int, QuestionDetails]) -> str:
    """
    Given a set of questions, generates a prompt for the question grouping task.
    """

    questions_dict = extract_questions(question_details_dict)
    TEMPLATE_UNIFY_QUESTIONS = f"""
    We have a dictionary with a set of questions. The keys are the question IDs and the values are the question themselves.

    Your task is to group the questions that are extremely related to each other.
    We will consider that this is the case when the questions are asking about the same topic, but from a different angle.
    Questions that are unrelated should be in their own group.

    Provide your answer in the following JSON format:

    {{
    "group_name": [question_id1, question_id2, ...],
    "group_name2": [question_id3],
    "group_name3": [question_id4, question_id5, ...],
    ...
    }}


    Here is the dictionary of questions:
    ```
    {questions_dict}
    ```
    """
    return TEMPLATE_UNIFY_QUESTIONS


# TODO: FUNCIÓN PARA ENVIAR EL PROMPT DE AGRUPAMIENTO AL LLM


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
        background=question_details.description,
        resolution_criteria=question_details.resolution_criteria,
        fine_print=question_details.fine_print,
    )


def apply_template_for_details_unification(question_details_dict: Dict[int, QuestionDetails], question_ids: Iterable[int]) -> str:
    """
    Generates a prompt for unifying the details of similar questions.

    Parameters:
    - question_details_dict (Dict[int, QuestionDetails]): A dictionary where keys are question IDs and values are dictionaries containing question details.
    - question_ids (Iterable[int]): An iterable of question IDs to be unified. It is assumed that all the questions in the iterable are indeed similar.

    Returns:
    - str: A formatted string containing the prompt for unifying the details of the specified questions.
    """

    questions_details = [question_details_dict[q_id] for q_id in question_ids]
    question_str_list = [make_question_str(details) for details in questions_details]
    question_str = "\n".join(question_str_list)

    UNIFICATION_QUERY = f"""
    You will recieve a series of similar questions. Each question has it's own background information and resolution criteria, even though they might be very similar.

    Your task is to synthesize the information from all the questions in a group and provide a unified background and resolution criteria for the group.
    You will do this by deleting information that is repeated in all the questions, and keeping information that is different between the questions.
    Your task is only to eliminate redundancies, no information should be lost in this process.

    It is okay for the unified question to now be a series of different questions, though grouped together.

    Provide your answer in the following JSON format:

    {{
        "unified_title": "string",
        "unified_background": "string",
        "unified_resolution_criteria": "string",
        "unified_fine_print": "string",
        "unified_question": "string"
    }}

    {question_str}
    """

    return UNIFICATION_QUERY


# TODO: FUNCIÓN PARA ENVIAR EL PROMPT DE UNIFICACIÓN AL LLM


def apply_question_template_to_unification_json(question_ids: Iterable[int],
                                                question_details_dict: Dict[int, QuestionDetails],
                                                unification_dict: Dict[str, str]) -> str:
    """
    Generates a unified string with the details of the questions to be forecasted.

    Parameters:
    - question_ids (Iterable[int]): An iterable of the question IDs that were unified.
    - question_details_dict (Dict[int, Dict]): A dictionary where keys are question IDs and values are dictionaries containing question details.
    - unification_dict (Dict[str, str]): A dictionary containing the LLM's output for the task of unifying the question details.

    Returns:
    - str: A formatted string containing the unified details of the questions to be forecasted.
    """

    question_str = collapse_questions_into_str(question_ids, question_details_dict)

    QUESTION_TEMPLATE = """
Title: "{title}"

{question_str}

The Resolution Criteria is:
```
{resolution_criteria}
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

    return QUESTION_TEMPLATE.format(
        question_str=question_str,
        title=unification_dict["unified_title"],
        background=unification_dict["unified_background"],
        resolution_criteria=unification_dict["unified_resolution_criteria"],
        fine_print=unification_dict["unified_fine_print"],
    )
