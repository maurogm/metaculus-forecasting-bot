from src.metaculus import get_prediction

def main():
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Who won the world series in 2020?"},
        {"role": "assistant", "content": "The Los Angeles Dodgers won the World Series in 2020."},
        {"role": "user", "content": "Where was it played?"}
    ]
    response = get_prediction("gpt-3.5-turbo", messages)
    print(response)

if __name__ == "__main__":
    main()
