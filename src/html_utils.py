from lxml.html.clean import Cleaner
from bs4 import BeautifulSoup, FeatureNotFound
import re
import requests
from typing import List


def extract_urls(text: str) -> List[str]:
    """
    Extracts all URLs from the given text.

    Parameters:
        text (str): The text from which to extract URLs.

    Returns:
        List[str]: A list of URLs found in the text.
    """
    return re.findall(r'https?://[^\s\)\]\}\>\,]+', text)


def fetch_html(url: str) -> str:
    """
    Fetches the HTML content of the given URL.

    Parameters:
        url (str): The URL from which to fetch HTML content.

    Returns:
        str: The HTML content fetched from the URL.

    Raises:
        Exception: If an error occurs during the HTTP request.
    """
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()  # Verify that the request was successful
    except requests.exceptions.RequestException as e:
        raise Exception(f"Error fetching {url}: {e}")
    return response.text


def clean_html_content(html: str) -> str:
    """
    Cleans the given HTML content by removing scripts, comments, styles, and other unwanted elements.

    Parameters:
        html (str): The raw HTML content to be cleaned.

    Returns:
        str: The cleaned HTML content.

    Raises:
        Exception: If an error occurs during HTML cleaning.
    """
    cleaner = Cleaner(
        scripts=True,
        javascript=True,
        comments=True,
        style=True,
        links=False,
        meta=False,
        page_structure=False,
        processing_instructions=True,
        embedded=True,
        frames=True,
        forms=True,
        annoying_tags=True,
        safe_attrs_only=True
    )
    try:
        clean_html = cleaner.clean_html(html)
    except Exception as e:
        raise Exception(f"An error occurred while cleaning HTML content: {e}")
    return clean_html


def strip_html_tags(html_content: str) -> str:
    """
    Strips HTML tags from the given HTML content and returns the text content.

    Parameters:
        html_content (str): The HTML content from which to extract text.

    Returns:
        str: The extracted text content without HTML tags.

    Raises:
        Exception: If an error occurs during HTML parsing.
    """
    try:
        soup = BeautifulSoup(html_content, "lxml")
        text = soup.get_text(separator='\n').strip()
    except FeatureNotFound:
        raise Exception(
            "The 'lxml' parser is not available. Please install it or choose a different parser.")
    except Exception as e:
        raise Exception(f"An error occurred while stripping HTML tags: {e}")
    return text


def clean_whitespace(text: str) -> str:
    """
    Cleans up whitespace in the given text by replacing multiple spaces and tabs with a single space,
    and multiple newlines with a single newline.

    Parameters:
        text (str): The text content to be cleaned.

    Returns:
        str: The text content with cleaned whitespace.
    """
    try:
        text = re.sub(r'[ \t]+', ' ', text)
        text = re.sub(r'\n+', '\n', text)
    except re.error as e:
        raise Exception(f"An error occurred while cleaning whitespace: {e}")
    return text.strip()


def clean_html(html: str) -> str:
    """
    Cleans the given HTML content and extracts the text content with cleaned whitespace.

    This function performs the following steps:
    1. Cleans the HTML content using `clean_html_content`.
    2. Strips HTML tags using `strip_html_tags`.
    3. Cleans up whitespace using `clean_whitespace`.

    Parameters:
        html (str): The raw HTML content to be processed.

    Returns:
        str: The final cleaned text content.
    """
    clean_html = clean_html_content(html)
    stripped_text = strip_html_tags(clean_html)
    cleaned_text = clean_whitespace(stripped_text)
    return cleaned_text
