from asknews_sdk import AskNewsSDK
from src.config import ASKNEWS_CLIENT_ID, ASKNEWS_CLIENT_SECRET, logger_factory

from dataclasses import dataclass, field
from typing import Optional, Any
from logging import Logger

@dataclass
class AskNewsFetcher:
    """
    Class to handle the fetching of news articles from the AskNews API.

    Parameters
    ----------
        n_hot_articles : int
            Number of hot articles to fetch. Default is 10.
        n_historical_articles : int
            Number of historical articles to fetch. Default is 20.
        query : str
            Query to search for.

    Attributes
    ----------
        hot_response : None
            Response from the hot articles search.
        historical_response : None
            Response from the historical articles search.


    Methods
    -------
    fetch_articles()
        Fetches the articles from the AskNews API.
    make_news_str()
        Generates a string with the news articles.

    Examples
    --------
    >>> news_fetcher = AskNewsFetcher("Covid-19, hot_articles=5, historical_articles=20)")
    >>> news_fetcher.fetch_articles()

    """
    query: str
    n_hot_articles: int = 10
    n_historical_articles: int = 20
    hot_response: Optional[Any] = None
    historical_response: Optional[Any] = None
    logger: Logger = field(init=False, default=None)

    def __post_init__(self):
        self.logger = logger_factory.make_logger(name="NewsFetcher")

    def fetch_articles(self):
        ask = AskNewsSDK(
            client_id=ASKNEWS_CLIENT_ID,
            client_secret=ASKNEWS_CLIENT_SECRET,
            scopes=["news"]
        )

        if self.query != "" and self.hot_response is None and self.n_hot_articles > 0:
            self.logger.debug(f"Fetching hot articles for query: {self.query}")
            self.hot_response = ask.news.search_news(
                query=self.query,
                n_articles=self.n_hot_articles,
                return_type="both",
                diversify_sources=True,
                strategy="latest news"
            )

        if self.query != "" and self.historical_response is None and self.n_historical_articles > 0:
            self.logger.debug(f"Fetching historical articles for query: {self.query}")
            self.historical_response = ask.news.search_news(
                query=self.query,
                n_articles=self.n_historical_articles,
                return_type="both",
                diversify_sources=True,
                strategy="news knowledge"
            )

    def make_news_str(self):

        if self.hot_response is None:
            hot_str = ""
        else:
            hot_str = f"\nThe following news articles provide current information on the topic: \n```{self.hot_response.as_string}```\n"

        if self.historical_response is None:
            historical_str = ""
        else:
            historical_str = f"\nThe following news articles give more historical context: \n```{self.historical_response.as_string}```\n"

        return hot_str + historical_str
