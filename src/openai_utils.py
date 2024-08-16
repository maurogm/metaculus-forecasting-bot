import json
import requests
from src.config import METACULUS_TOKEN, METACULUS_OPENAI_PROXY_URL
from src.data_models.CompletionResponse import CompletionResponse

import os
from typing import List, Dict, Any


def get_gpt_prediction_via_proxy(messages: List[Dict[str, str]], model: str = "gpt-4o") -> CompletionResponse:
    """
    Request a prediction using the OpenAI API through the Metaculus proxy.

    This function sends a request to the Metaculus proxy to make a prediction using the OpenAI API.
    The proxy is used in order to use the credits granted by OpenAI for the competition.

    Args:
        messages (List[Dict[str, str]]): `messages` parameter to be forwarded to the OpenAI API.
        model (str): OpenAI model to be used.

    Returns:
        Dict[str, Any]: Forwarded response of the OpenAI API.
    """
    if METACULUS_TOKEN is None:
        raise ValueError(
            "The environment variable METACULUS_TOKEN is not set.")

    if METACULUS_OPENAI_PROXY_URL is None:
        raise ValueError(
            "The environment variable METACULUS_OPENAI_PROXY_URL is not set.")

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Token {METACULUS_TOKEN}"
    }

    data_request = {
        "model": model,
        "messages": messages
    }

    response = requests.post(METACULUS_OPENAI_PROXY_URL,
                             headers=headers, data=json.dumps(data_request))
    response.raise_for_status()

    # gpt_text = response.json()["choices"][0]["message"]["content"]
    return CompletionResponse(response.json())


def collapse_messages_into_string(messages: List[Dict[str, str]]) -> str:
    """
    Collapses a list of messages into a single string.

    Useful to make strings that can be pasted into ChatGPT.

    Parameters:
    - messages (List[Dict[str, str]]): A list of messages to be collapsed.

    Returns:
    - str: A single string containing all the messages.
    """
    contents = [msg["content"] for msg in messages]
    return "\n".join(contents)

