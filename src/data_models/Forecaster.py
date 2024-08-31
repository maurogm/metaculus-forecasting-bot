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

from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnablePassthrough, RunnableSequence
from src.openai_utils import make_proxied_ChatOpenAI_LLM
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_community.callbacks.manager import get_openai_callback

llm = make_proxied_ChatOpenAI_LLM(temperature=0.1)


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
You are a member of a team of forecasters.
You are trying to come up with a forecast for one or more questions.
To create an accurate forecast, your team follows a meticulous series of steps.
You will perform one of these steps, for which you will receive the instructions.
You are thorough and precise in your responses.
Your answers are concise and to the point.

{question_details}
"""

prompt_str_extract_key_details = """
Step 1: **Extract Key Information from the Question Details**:
   - Identify and list the key elements and variables from the question and background information.
   - Note any important dates, events, or factors that may influence the outcome.
   - If there are multiple questions, explicitly note if they are correlated in any way, such as being mutually exclusive.
"""
prompt_str_ask_questions = """
Step 2: **Ask yourself questions**:
   - If you feel that there are pieces of information that you are missing, ask yourself questions that will help you fill in the gaps.
   - If there are events unfolding that could affect the outcome, ask yourself questions about those events.
   - If thereare things still to come that could affect the outcome, ask yourself questions about those things.
Provide your answer as a bullet list.
"""
prompt_str_extract_info_from_news = """
Today is {today}.

You are given the following pieces of news articles:
{news_articles}

Step 3: **Extract Relevant Insights from the News Articles**:
   - Be mindful of both quantitative and qualitative information.
   - When extracting information, consider the following questions, but also any other things that you deem relevant:
   - What's the current state of affairs?
   - Have there been any recent developments that could impact the outcome?
   - Has the situation changed between the dates of different news articles?
   - Is there a consensus among the sources? If not, what are the conflicting views?
   - Do the news articles provide any data or numbers that can be used to inform the forecast?
   - Are there any expert opinions or statements that could be relevant?
   - Does the path ahead appear to be clear or are there uncertainties?

Provide your answer as a bullet list of facts and insights that you extracted from your analysis.
"""
prompt_str_build_timeline = """

Step 4: Build Yourself a Timeline with every relevant event that you have identified. For example:
   - Today is {today}.
   - What is the resolution date?
   - What significant events have happened in the past that might be related to the matter at hand?
   - Are there any significant events expected to happen between now and the resolution that might have an impact?
   - Add to your timeline anything that you deem relevant.

You are provided the following report as a complementary data source:
{news_insights}

Provide your answer as a bullet list of times and events. Be concise. Include only dated events.
"""
prompt_str_preliminar_assessment = """

You are provided the following report as a complementary data source:
```markdown
{news_insights}
```

## **Historical Trends Analysis**:
   - Look for historical data or trends related to the question.
   - Determine if there are any patterns or cycles that can inform the forecast.
   - Take note of past reference values and events, both in the extremes and averages.
   - When appropriate, have an idea of the range of values that the variable can take, and the variability of the variable.

## **Current Events Evaluation**:
   - Today is {today}.
   - Assess the current situation and recent developments that could impact the forecast.
   - Factor in any recent news, changes in policies, or significant events.
   - Decide if the present situation is in line with historical trends or if there are deviations.

"""
prompt_str_baseline_scenario = """

You are provided the following report assessing the current situation and some historical trends:
```markdown
{preliminar_assessment}
```

## Define a Baseline Scenario. Some topics that might help you settle on your baseline might include:
   - Is there a baseline scenario? Or does this line of thinking not apply to this situation?
   - What has happened in the past in similar situations?
   - If by the resolution date nothing has changed from the present situation, how would the question resolve?
   - How drastic a change would have to happen in order to modify that? Consider current levels and trends (and seasonality, if applicable).
   - Does the time left until the resolution (as seen in Step 4) seem enough for those changes to occur?
   - Could such a change happen naturally, or would it take a rare triggering event? If so, has such a rare triggering event happened in the past? How frequently? Does the current context lead you to think that the likelihood of the triggering event is substantially modified, or is it better to keep yourself aligned with the historical base rate?
   - If an event is dramatic and has few precedents, then the baseline should be extremely low: don't be afraid to assign a very low probability to such events. For example, sudden regime changes, unforseen and exceptional natural disasters, unexpected and sudden deaths of public figures in a short time period, the invasion of a country by another, the detonation of nuclear weapons, etc. are events that are extremely rare and should be assigned an extremely low probability (even as low as 1%) under normal circumstances.
   - On the other hand, situations that are stable and have been stable for a long time, and that have a lot of precedents, should be assigned a high probability. For example, the sun rising tomorrow, the fact that the vast majority of people will not die in the next 24 hours, that stable democracies will continue to be so for the next couple of years, the USA having the largest GDP in the world, should all have extremely high probabilities.


Provide your baseline in a concise manner, as well as some reasoning behind it.
You don't need to redundantly repeat the information that was already given to you, but you can refer to it.
"""
prompt_str_initial_predictions = """


Step 8: **Make Predictions**:
   - Based on the analysis, make an initial prediction.
   - You must write it down with the "Initial forecast: [your prediction]".
   - Consider the probability of different outcomes and articulate the reasoning behind your prediction.

To give you a place to start from, an assistant provided you with the following baseline scenario:
```
{baseline_scenario}
```

Another assistant provided the following report assessing the current situation and some historical trends:
```
{preliminar_assessment}
```

For each question, provide your initial forecast as a single number between 0.01 and 0.99.
"""


prompt_str_baseline_and_prediction_scenario = """
You are provided the following report assessing the current situation and some historical trends:
```
{preliminar_assessment}
```

## Define a Baseline Scenario. Some topics that might help you settle on your baseline might include:
   - Is there a baseline scenario? Or does this line of thinking not apply to this situation?
   - What has happened in the past in similar situations?
   - If by the resolution date nothing has changed from the present situation, how would the question resolve?
   - How drastic a change would have to happen in order to modify that? Consider current levels and trends (and seasonality, if applicable).
   - Does the time left until the resolution (as seen in Step 4) seem enough for those changes to occur?
   - Could such a change happen naturally, or would it take a rare triggering event? If so, has such a rare triggering event happened in the past? How frequently? Does the current context lead you to think that the likelihood of the triggering event is substantially modified, or is it better to keep yourself aligned with the historical base rate?
   - If an event is dramatic and has few precedents, then the baseline should be extremely low: don't be afraid to assign a very low probability to such events. For example, sudden regime changes, unforseen and exceptional natural disasters, unexpected and sudden deaths of public figures in a short time period, the invasion of a country by another, the detonation of nuclear weapons, etc. are events that are extremely rare and should be assigned an extremely low probability (even as low as 1%) under normal circumstances.
   - On the other hand, situations that are stable and have been stable for a long time, and that have a lot of precedents, should be assigned a high probability. For example, the sun rising tomorrow, the fact that the vast majority of people will not die in the next 24 hours, that stable democracies will continue to be so for the next couple of years, the USA having the largest GDP in the world, should all have extremely high probabilities.


Provide your baseline in a concise manner, as well as some reasoning behind it.
You don't need to redundantly repeat the information that was already given to you, but you can refer to it.

## **Make Predictions**:
   - Based on the analysis, make an initial prediction.
   - You should rely heavily on the baseline scenario and the historical trends, but can also weigh in the current situation and recent developments.
   - Consider the probability of different outcomes and articulate the reasoning behind your prediction.

For each question, provide your initial forecast as a single number between 0.01 and 0.99.
"""


prompt_str_check_predictions_implications = """

You are provided the following report assessing the current situation and some historical trends:
```markdown
{preliminar_assessment}
```

You have made an initial forecast:
```
{baseline_prediction}
```

## **Cross-Check Against Historical Frequency**
    Reflect on your forecast and consider the broader historical context:
        1. Historical Baseline: How often has the event you are forecasting actually occurred in similar contexts in the past? Quantify this frequency if possible.
        2. What frequency is implied by your forecast?
          - If time is a factor, consider how much time is left from today ({today}) until the resolution date.
        2. Consistency Check: Compare your current probability estimate with this historical baseline:
         - If your forecasted probability implies that the event should occur more frequently than it historically has, consider whether there is a clear and justifiable reason for this discrepancy.
         - Conversely, if your forecasted probability is lower than the historical frequency, assess whether current conditions are significantly more stable than in the past.
        3. Re-Evaluate: If your current probability estimate is much higher or lower than what historical data would suggest, re-examine your reasoning. Is there something unique about the current context that justifies this difference? Or does the historical baseline suggest you should adjust your probability closer to the historical average?
"""
prompt_str_assess_uncertainty = """

You are provided the following report assessing the current situation and some historical trends:
```
{preliminar_assessment}
```


A coleague has made the following prediction:
```
{initial_predictions}
```

You however wonder if that forecast is correctly taking into account the uncertainty of the situation.
   - Identify potential sources of uncertainty or factors that could alter the forecast.
   - Quantify the uncertainty where possible and explain how it affects the confidence in the prediction.

Give your opinion on the uncertainty of the forecast and provide a concise explanation.
If you agree with the forecast, just say so.
But if you disagree, explain why, and say how you would adjust the forecast.
"""
prompt_str_review_and_refine = """
Final Step: **Review and Refine**.
You are in charge of making the final forecast.

You are provided the following report assessing the current situation and some historical trends:
```
{preliminar_assessment}
```

An assistant provided you with the following baseline scenario and prediction:
```
{baseline_prediction}
```

Regarding that prediction, a colleague has checked it against historical frequency and made the following assessment:
```
{check_predictions_implications}
```   

Review the initial forecast and ensure it aligns with the data and analysis.
  - Consider alternative scenarios or viewpoints that could lead to different outcomes from the baseline.
  - Identify potential sources of uncertainty or factors that could alter the forecast.
  - Quantify the uncertainty where possible and explain how it affects your confidence in the prediction.
  - Ponder your colleagues' assesment regarding the historical frequency of the event.

It is very important to check the following points:
  - If you are forecasting mutually exclusive events, ensure their probabilities sum to 1.
  - If multiple questions are correlated, check that the forecasts are consistent with each other. If they are, explain why. If they are not, refine your forecasts.

Refine your prediction if necessary, to improve accuracy and clarity. Finally, write down your final forecast.
"""
prompt_str_json_output = """
The revised and final forecast is as follows:
{final_forecast}

{output_instructions}
"""

def make_chat_prompt_template(prompt_str):
    return ChatPromptTemplate([("system", system_str), ("user", prompt_str)])


prompt_template_extract_key_details = make_chat_prompt_template(prompt_str_extract_key_details)
prompt_template_ask_questions = make_chat_prompt_template(prompt_str_ask_questions)
prompt_template_extract_info_from_news = make_chat_prompt_template(prompt_str_extract_info_from_news)
prompt_template_preliminar_assessment = make_chat_prompt_template(prompt_str_preliminar_assessment)
prompt_template_baseline_scenario = make_chat_prompt_template(prompt_str_baseline_scenario)
prompt_template_initial_predictions = make_chat_prompt_template(prompt_str_initial_predictions)
prompt_template_baseline_and_prediction_scenario = make_chat_prompt_template(prompt_str_baseline_and_prediction_scenario)
prompt_template_check_predictions_implications = make_chat_prompt_template(prompt_str_check_predictions_implications)
prompt_template_assess_uncertainty = make_chat_prompt_template(prompt_str_assess_uncertainty)
prompt_template_review_and_refine = make_chat_prompt_template(prompt_str_review_and_refine)
prompt_template_json_output = make_chat_prompt_template(prompt_str_json_output)



def extract_details_str(details_preparation: DetailsPreparation) -> str:
    return details_preparation.make_details_str()


chain_make_details_str = RunnableLambda(extract_details_str)
chain_extract_key_details = prompt_template_extract_key_details | llm | StrOutputParser()
chain_ask_questions = prompt_template_ask_questions | llm | StrOutputParser()
chain_extract_info_from_news = prompt_template_extract_info_from_news | llm | StrOutputParser()
def news_route_function(input):
    maybe_news = input.get("news_object")
    if maybe_news is not None and isinstance(maybe_news, AskNewsFetcher):
        news_str: str = maybe_news.make_news_str()
        return (
            RunnablePassthrough.assign(news_articles=RunnableLambda(lambda x: news_str)) |
            RunnablePassthrough.assign(news_insights=chain_extract_info_from_news))
    else:
        return RunnablePassthrough.assign(news_insights=RunnableLambda(lambda x: "No news provided."))
chain_news_route = RunnableLambda(news_route_function)
# chain_build_timeline = prompt_template_build_timeline | llm | StrOutputParser()
chain_preliminar_assessment = prompt_template_preliminar_assessment | llm | StrOutputParser()
chain_baseline_scenario = prompt_template_baseline_scenario | llm | StrOutputParser()
chain_initial_predictions = prompt_template_initial_predictions | llm | StrOutputParser()
chain_baseline_and_prediction_scenario = prompt_template_baseline_and_prediction_scenario | llm | StrOutputParser()
chain_check_predictions_implications = prompt_template_check_predictions_implications | llm | StrOutputParser()
chain_assess_uncertainty = prompt_template_assess_uncertainty | llm | StrOutputParser()
chain_review_and_refine = prompt_template_review_and_refine | llm | StrOutputParser()
chain_json_output = prompt_template_json_output | llm | JsonOutputParser()

output_instructions = """
Your answer MUST consist of a JSON with the following format:
{
    "forecasts": {{question_id: forecast}}, # each forecast is a float between 0 and 1 representing the probability of the event occurring
    "summaries": {{question_id: summary}} # summary should be a long paragraph highlighting the key points of your reasoning that led to the forecast
}
"""

chain_full = (
    RunnablePassthrough.assign(output_instructions=RunnableLambda(lambda x: output_instructions)) |
    RunnablePassthrough.assign(news_insights=chain_news_route) |
    RunnablePassthrough.assign(preliminar_assessment=chain_preliminar_assessment) |
    RunnablePassthrough.assign(baseline_prediction=chain_baseline_and_prediction_scenario) |
    RunnablePassthrough.assign(check_predictions_implications=chain_check_predictions_implications) |
    RunnablePassthrough.assign(final_forecast=chain_review_and_refine) |
    RunnablePassthrough.assign(json_output=chain_json_output)
)