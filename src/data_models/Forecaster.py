import json
import datetime

from typing import Dict, Iterable, List, Optional, Any

from src.config import OPENAI_MODEL, logger_factory

from src.data_models.DetailsPreparation import DetailsPreparation
from src.data_models.AskNewsFetcher import AskNewsFetcher
from src.data_models.CompletionResponse import CompletionResponse

from src.openai_utils import get_gpt_prediction_via_proxy
from src.utils import trim_beginning_of_string, try_to_find_and_eval_dict

from dataclasses import dataclass, field
from logging import Logger


@dataclass
class Forecaster:
    """
    Class to handle the forecasting of questions.

    This class provides methods to fetch, parse, and persist the forecast response.

    Parameters
    ----------
    details_preparator : DetailsPreparation
        Instance of DetailsPreparation containing the details of the questions to forecast.
    news : Optional[AskNewsFetcher], optional
        Instance of AskNewsFetcher containing the news context.
    forecast_response : Optional[CompletionResponse], optional
        Instance of CompletionResponse containing the forecast response.
    forecast_dict : Optional[Dict[str, Any]], optional
        Dictionary containing the parsed forecast response.
    logger : Any, optional
        Logger instance.

    Examples
    --------
    >>> forecaster = Forecaster(details_preparator, news)
    >>> forecaster.fetch_forecast_response()
    >>> forecaster.parse_forecast_response()
    >>> forecaster.persist_forecast()
    """

    details_preparator: DetailsPreparation
    news: Optional[AskNewsFetcher] = None
    forecast_response: Optional[CompletionResponse] = None
    forecast_dict: Optional[Dict[str, Any]] = None
    logger: Logger = field(init=False, default=None)

    def __post_init__(self):
        self.logger = logger_factory.make_logger(name="Forecaster")
    
    def fetch_forecast_response(self) -> None:
        self.logger.debug(f"Fetching forecast response for question IDs {self.__q_ids_str}")
        if self.forecast_response is not None:
            self.logger.warning(
                "Tried to fetch forecast response when it was already fetched.")
        else:
            messages = self.make_messages_for_forecast()
            self.forecast_response = get_gpt_prediction_via_proxy(
                messages, model=OPENAI_MODEL)

    def make_messages_for_forecast(self) -> List[Dict[str, str]]:
        question_details_str = self.details_preparator.make_details_str()
        if self.news is None:
            messages = make_messages_for_forecaster(question_details_str)
        else:
            news_context_str = self.news.make_news_str()
            messages = make_messages_for_forecaster(question_details_str, news_context_str)
        return messages
    
    def parse_forecast_response(self):
        trimmed_answer = trim_beginning_of_string(self.forecast_response.content, "answer_dictionary")
        try:
            parsed_dict = try_to_find_and_eval_dict(trimmed_answer)
            # Let's make sure that the types are as expected:
            sanitized_forecasts = {int(k): float(v) for k, v in parsed_dict["forecasts"].items()}
            sanitized_summaries = {int(k): v for k, v in parsed_dict["summaries"].items()}
            self.forecast_dict = {"forecasts": sanitized_forecasts, "summaries": sanitized_summaries}
        except:
            self.logger.error(f"Tried to evaluate this string failed:\n{trimmed_answer}\n")
            raise ValueError("Failed to parse forecast response content.")
    
    def persist_forecast(self, path_to_dir: str = "logs/forecasts"):
        filename = f"{path_to_dir}/{self.__q_ids_str}.md"
        
        usage_str = f"#Usage:\n{self.forecast_response.tokens_all}"
        input_str = f"#Input:\n{self.make_messages_for_forecast()}"
        output_str = f"#Forecast response:\n{self.forecast_response.content}"

        with open(filename, "w") as f:
            f.write(f"{usage_str}\n{input_str}\n{output_str}")
    
    @property
    def __q_ids_str(self):
        q_ids = self.details_preparator.question_ids
        q_ids.sort()
        return "_".join([str(q_id) for q_id in q_ids])



def make_messages_for_forecaster(question_str: str, news_str: Optional[str] = None):
    system_message = f"{system_str}\n{forecasting_chatgpt_instructions_str}\n{answer_format_str}"
    messages = [{"role": "system", "content": system_message}]
    if news_str is not None:
        messages.append({"role": "assistant", "content": news_str})
    messages.append({"role": "user", "content": question_str})
    return messages

today = datetime.datetime.now().strftime("%Y-%m-%d")

system_str = """
You are a professional forecaster.
You will recieve one or more questions, and your job is to provide a forecast for each question.
The forecast should be the probability that the event described in the question will occur.
Format your answers as markdown text.
"""

forecast_examples_str = """
This probabilities should aknowledge the uncertainty of the future, but also relay heavily into historical data and trends. Here are a list of examples of how this aspects might come into play:
####
- "Will the sun rise tomorrow?" should have a probability of 100%.
- "Will Canada continue to be a democracy in 2025?" should have a probability of 99%.
- "Will a nuclear weapon be detonated next month?" should have a probability of 1%, since this has not happend since the end of WWII, and the consequences of such an event would be world-changing. 
- "Will an AGI be created next month?" should have a probability of 0.1%, since this is a very hard problem whose very solution is speculation, and would also be world-changing.
- "Will China invade Taiwan in the next year?" should have a probability of less than 4%, since despite this being a very complex geopolitical problem, they have been stable for decades, and such an action would have world-changing consequences.
- "Which among X and Y will win the next Eurocup final?" should have a probability between 40% and 60%, since the two teams are probably evenly matched to be in the finals, and there is a lot of luck involved.
- "Will X young and healthy person die in the next month?" should have a probability of less than 1%, since that happening in such a short time frame would be a great coincidence.
- "Will the USA continue to have the largest GDP in the world in 2025?" should have a probability of 98%, since this has been the case for decades, and there is not enough time for someone else to surpass them.
####
Of course, these are baseline probabilities, and you should adjust them according to the specific context of the question. But don't move too much from the premise that the future is uncertain, that the past is the best predictor of the future, and that world-changing happen only VERY unfrequently.
""" # NOTE: This is not being used in the current implementation.

forecasting_chatgpt_instructions_str = f"""
To create an accurate forecast, please follow these steps:

Step 1: **Extract Key Information from the Question Details**:
   - Identify and list the key elements and variables from the question and background information.
   - Note any important dates, events, or factors that may influence the outcome.
   - If there are multiple questions, explicitly note if they correlated in any way, such as being mutially exclusive.

Step 2: **Ask yourself questions**:
   - If you feel that there are pieces of information that you are missing, ask yourself questions that will help you fill in the gaps.
   - If there are events unfolding that could affect the outcome, ask yourself questions about those events.
   - If thereare things still to come that could affect the outcome, ask yourself questions about those things.

Step 3: **Extract Relevant Insights from the News articles**:
   - Be mindful of both quantitative and qualitative information.
   - When extracting information consider the following questions, but also any other thing that you deem relevant:
   - What's the current state of affairs?
   - Have there been any recent developments that could impact the outcome?
   - Has the sitauation changed between the dates of different news articles?
   - Is there a consensus among the sources? If not, what are the conflicting views?
   - Do the news articles provide any data or numbers that can be used to inform the forecast?
   - Are there any expert opinions or statements that could be relevant?
   - Does the path ahead appear to be clear or are there uncertainties?
   - Include any information that could shed light on the questions you asked yourself in Step 2.

Step 4: **Build yourself a timeline**:
   - Today is {today}.
   - What is the resolution date?
   - What significant events have happened in the past that might be related with the matter at hand?
   - Are there any significant events expected to happen between now and the resolution that might have an impact?
   - Add to your timeline anything that you deem relevant.

Step 5: **Analyze Historical Trends**:
   - Look for historical data or trends related to the question.
   - Determine if there are any patterns or cycles that can inform the forecast.
   - Take note of past reference values and events, both in the extremes and averages.
   - When appropriate, have an idea of the range of values that the variable can take, and the the variability of the variable.

Step 6: **Evaluate Current Events**:
   - Assess the current situation and recent developments that could impact the forecast.
   - Factor in any recent news, changes in policies, or significant events.
   - Decide if the present situation is in line with historical trends or if there are deviations.
   
Step 7: **Define a baseline scenario**:
   - Is there a baseline scenario? Or does this line of thinking not apply to this situation?
   - What has happened in the past in similar situations?
   - If by the resolution date nothing has changed from the present situation, how would the question resolve?
   - How drastical a change would have to happen in order to modify that? Consider current levels and trends (and seasonality, if applicable).
   - Does the time left until the resolution (as seen on step 4) seem enough for those changes to occur?
   - Could such a change happen naturally, or would it take a rare triggering event? If so, has such a rare triggering event happened in the past? How frequently? Does the current context lead to think that the likelihood of the triggering event is substantialy modified, or is it better to keep yourself aligned with the historical base rate?
   - If an event is dramatic and has few precedents, then the baseline should be extremely low: don't be afraid to assign a very low probability to such events. For example, sudden regime changes, unforseen and exceptional natural disasters, unexpected and sudden deaths of public figures in a short time period, the invasion of a country by another, the detonation of nuclear weapons, etc. are events that are extremely rare and should be assigned an extremely low probability (even as low as 1%) under normal circumstances.
   - On the other hand, situations that are stable and have been stable for a long time, and that have a lot of precedents, should be assigned a high probability. For example, the sun rising tomorrow, the fact that the vast majority of people will not die in the next 24 hours, that stable democracies will continue to be so for the next couple of years, the USA having the largest GDP in the world, should all have extremely high probabilities.

Step 8: **Make Predictions**:
   - Based on the analysis, make an initial prediction and write it down.
   - Consider the probability of different outcomes and articulate the reasoning behind your prediction.

Step 9: **Ponder alternative viewpoints**:
   - Consider alternative scenarios or viewpoints that could lead to different outcomes.
   - Think about how different factors or events could change the forecast.

Step 10: **Assess Uncertainty**:
   - Identify potential sources of uncertainty or factors that could alter the forecast.
   - Quantify the uncertainty where possible and explain how it affects your confidence in the prediction.

Step 11: **Review and Refine**:
   - Review the forecast and ensure it aligns with the data and analysis.
   - You must explicitly consider each of the following points and write down your consideration, but you are not limited to them:
      - Consider if your initial asessment (Step 8) took into account the viewpoints and uncertainties you identified in steps 8 and 9.
      - If you are forecasting mutually exclusive events, ensure their probabilities sum to 1.
      - If multiple questions are correlated, check that the forecasts are consistent with each other. If they are, explain why. If they are not, refine your forecasts.
   - Refine your prediction if necessary, to improve accuracy and clarity.
   - Finally, write down your final forecast.

Step 12: **Document Reasoning**:
   - Clearly document each step of your reasoning process.
   - Provide a summary that explains how you arrived at the forecast, including key data points and considerations.
"""

answer_format_str = """
You should document your thougt process and reasoning while following the steps in the instructions.
Take as much time as you need every step of the way, and don't rush to make a prediction.

The end of your answer should contain a dictionary with the following format, preceded by a delimiter of four hashtags (i.e., ####), as follows:

####
answer_dictionary =
{
    "forecasts": {{question_id: forecast}}, # each forecast is a float between 0 and 1 representing the probability of the event occurring
    "summaries": {{question_id: summary}} # summary should be a long paragraph highlighting the key points of your reasoning that led to the forecast
}
"""
