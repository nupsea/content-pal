import json
import uuid
import argparse

import requests
import questionary

import pandas as pd


def get_random_query(file_path):
    df = pd.read_csv(file_path)
    return df.sample(n=1).iloc[0]["query"]


def ask_query(url, query):
    data = {"query": query}
    response = requests.post(url, json=data)
    return response.json()


def send_feedback(url, conversation_id, feedback):
    feedback_data = {"conversation_id": conversation_id, "feedback": feedback}
    response = requests.post(f"{url}/feedback", json=feedback_data)
    return response.status_code


def main():
    parser = argparse.ArgumentParser(
        description="Interactive Content Pal CLI app for continuous query answering and feedback"
    )
    parser.add_argument(
        "--random", action="store_true", help="Use random queries from the CSV file"
    )
    args = parser.parse_args()

    base_url = "http://localhost:5001"
    csv_file = "./data/ground_truth_retrieval.csv"

    print("Welcome to the interactive query-answering app!")
    print("You can exit the program at any time when prompted.")

    while True:
        if args.random:
            query = get_random_query(csv_file)
            print(f"\nRandom query: {query}")
        else:
            query = questionary.text("Enter your watch-content query:").ask()

        response = ask_query(f"{base_url}/recommend", query)
        print("\nAnswer:", response.get("answer", "No answer provided"))

        conversation_id = response.get("conversation_id", str(uuid.uuid4()))

        feedback = questionary.select(
            "How would you rate this response?",
            choices=["+1 (Positive)", "-1 (Negative)", "Pass (Skip feedback)"],
        ).ask()

        if feedback != "Pass (Skip feedback)":
            feedback_value = 1 if feedback == "+1 (Positive)" else -1
            status = send_feedback(base_url, conversation_id, feedback_value)
            print(f"Feedback sent. Status code: {status}")
        else:
            print("Feedback skipped.")

        continue_prompt = questionary.confirm("Do you want to continue?").ask()
        if not continue_prompt:
            print("Thanks for using the app. Goodbye!")
            break


if __name__ == "__main__":
    main()