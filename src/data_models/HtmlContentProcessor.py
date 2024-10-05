import json
from typing import Dict, List

import itertools
from src.html_utils import fetch_html, extract_urls, clean_html
from src.config import logger_factory, llm_smart
from src.data_models.DetailsPreparation import DetailsPreparation
from src.openai_utils import make_proxied_ChatOpenAI_LLM

from langchain import LLMChain
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.callbacks.manager import get_openai_callback





class HtmlContentProcessor:
    """
    Class to handle fetching, cleaning, and processing HTML content from URLs extracted from a DetailsPreparation object.

    This class performs the following steps:
    - Extracts URLs from the 'background' field of the DetailsPreparation object's unified details.
    - For each URL:
        - Fetches and cleans the HTML content.
        - Applies an LLM call to extract important information.

    Parameters
    ----------
    details_preparation : DetailsPreparation
        The DetailsPreparation object containing the unified details from which to extract URLs.

    Attributes
    ----------
    details_preparation : DetailsPreparation
        The DetailsPreparation object provided.
    llm_responses : Dict[str, str]
        Dictionary mapping each URL to the LLM's response.

    Methods
    -------
    run()
        Runs the entire processing pipeline.
    """

    def __init__(self, details_preparation: 'DetailsPreparation'):
        self.logger = logger_factory.make_logger(name="HtmlContentProcessor")
        self.details_preparation = details_preparation
        self.llm_responses: Dict[str, str] = {}
        self.question_details_str = self.details_preparation.make_details_str()

    def run(self):
        """
        Runs the entire processing pipeline:
        - Extract URLs from the background information.
        - For each URL:
            - Fetch and clean HTML content.
            - Apply LLM to the cleaned text.
        """
        self.logger.debug("Starting processing pipeline.")
        urls = self.extract_urls_from_backgrounds()
        for url in urls:
            try:
                self.logger.debug(f"Processing URL: {url}")
                raw_html = fetch_html(url)
                if raw_html:
                    clean_text = clean_html(raw_html)
                    llm_response = self.apply_llm_to_text(clean_text, url)
                    self.llm_responses[url] = llm_response
                else:
                    self.logger.warning(f"No content fetched from URL: {url}")
            except Exception as e:
                self.logger.error(f"Error processing URL {url}: {e}")
        self.logger.debug("Processing pipeline completed.")

    def extract_urls_from_backgrounds(self) -> List[str]:
        """
        Extracts URLs from the 'background' field of each QuestionDetails object in the question_details_dict attribute.

        This method iterates over all values in the `question_details_dict` attribute, extracts URLs from the `background`
        field of each element using the `extract_urls` function, and removes duplicates.

        Returns
        -------
        List[str]
            A list of unique URLs extracted from all 'background' fields.
        """
        nested_urls = [extract_urls(details.background) for details in self.details_preparation.question_details_dict.values()]
        flattened_urls = list(itertools.chain.from_iterable(nested_urls))
        unique_urls = list(set(flattened_urls))
        return unique_urls

    def apply_llm_to_text(self, text: str, url: str) -> str:
        """
        Applies LLM call to the cleaned text to extract important information.

        Parameters
        ----------
        text : str
            The cleaned text to be processed by the LLM.
        url : str
            The URL associated with the text, used in the prompt.

        Returns
        -------
        str
            The response from the LLM.

        Raises
        ------
        Exception
            If an error occurs during the LLM call.
        """
        try:
            prompt_template = ChatPromptTemplate([("user", prompt_str)]) 
            chain = prompt_template | llm_smart | StrOutputParser()
            self.logger.debug(f"Sending LLM request for URL: {url}")
            input_dict = {"question_details": self.question_details_str,
                          "url": url,
                          "scraped_text": text,
                          }
            with get_openai_callback() as cb:
                response = chain.invoke(input_dict)
                cb_str = f"OpenAI Callback for parsing of {url}: \n{cb.__str__()}\n"
                self.logger.info(cb_str)
            self.logger.debug(f"Received LLM response for URL: {url}")
            return response
        except Exception as e:
            self.logger.error(f"Error applying LLM to text from URL {url}: {e}")
            raise

    def collapse_responses_in_single_str(self) -> str:
        """
        Collapses the LLM responses stored in the class into a single formatted string.

        This method takes the `llm_responses` attribute of the class, where each key is a URL and each value is the extracted information
        from that URL, and concatenates all the responses into a single string formatted for readability.
        """
        collapsed_text = ""
        for url, response in self.llm_responses.items():
            collapsed_text += f"Information extracted from {url}:\n{response}\n\n"
        return collapsed_text.strip()


prompt_str = """
You are an assistant to a team of forecasters.
You are trying to come up with a forecast for one or more questions.

{question_details}

------

The question definition includes a link to the {url} website.
An automatic tool scraped some text from that link, but it still has a lot of noise and non-relevant content.
Your task is to read the text and extract a bullet list of facts and information that is relevant to the forecast that your team has to make.
Just extract the information and report it. Be thorough in your summary, paying special attention to dates and numbers, when relevant.
Also be sure to differentiate facts from opinions.
Don't add anything extra, since that is a job for the senior members of your group, not you.

Here is the scraped text:
```
{scraped_text}
```
"""