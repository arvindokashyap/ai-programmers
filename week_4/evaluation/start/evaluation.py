import json
import os
from openai import OpenAI
from langsmith import Client
from langsmith.evaluation import evaluate

# Rocket's TLS-inspecting firewall needs the Windows cert store
try:
    import truststore
    truststore.inject_into_ssl()
except ImportError:
    pass

# Initialize the OpenAI client
client = OpenAI()

# LangSmith client (reads LANGCHAIN_API_KEY / LANGSMITH_API_KEY from the environment)
langsmith_client = Client()

# Dataset name in LangSmith
dataset_name = "news_dataset_class"

# System prompt for extraction
SYSTEM_PROMPT = """Extract information from the news into a dictionary.
The dictionary keys are company_name, date_of_transaction, amount, product_service, location.
Make sure the output is in proper JSON with double quotes around the keys and values.
For all dates, use the following format: mm-dd-yyyy."""

def make_call_to_llm(input):
    # Extract the input content from the dataset item
    user_content = input["news"] if isinstance(input, dict) else input

    # Create the message array for the API call
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content}
    ]

    # Call the OpenAI API
    response = client.chat.completions.create(
        model="gpt-4o-mini-2024-07-18",
        messages=messages,
        temperature=0
    )

    # Extract the output from the response
    output = response.choices[0].message.content

    return {"output": output}

# llm_result is the result of the make_call_to_llm call
# It contains the model's response that we'll compare against the expected output
def perform_eval(llm_result, dataset_item):
    try:
        # Parse the model's output
        llm_output = json.loads(llm_result.outputs['output'])

        # Parse the expected output
        expected_output = json.loads(dataset_item.outputs['output'])

        # Extract score from response
        # For a simpler implementation, let's manually calculate the score
        total_keys = len(expected_output)
        correct_keys = sum(1 for key in expected_output if key in llm_output and llm_output[key] == expected_output[key])
        score = correct_keys / total_keys if total_keys > 0 else 0

        return {"score": score}
    except json.JSONDecodeError:
        # Handle the case where JSON parsing fails
        return {"score": 0.0}

def ensure_dataset_exists():
    """Upload news_dataset.csv to LangSmith if the dataset is not there yet.

    The README walks through doing this in the LangSmith UI. Doing it here as well
    means the script is runnable end-to-end on a fresh account, and it is a no-op
    once the dataset exists.
    """
    if langsmith_client.has_dataset(dataset_name=dataset_name):
        print(f"Using existing LangSmith dataset '{dataset_name}'")
        return

    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, "news_dataset.csv")
    langsmith_client.upload_csv(
        csv_file=csv_path,
        input_keys=["news"],
        output_keys=["output"],
        name=dataset_name,
        description="News articles with expected structured extraction (company, date, amount, product, location)",
    )
    print(f"Uploaded {csv_path} to LangSmith as dataset '{dataset_name}'")

# Evaluate the target task
if __name__ == "__main__":
    ensure_dataset_exists()

    # Run make_call_to_llm over every example in the dataset and score each
    # prediction with perform_eval. LangSmith records the whole run as an
    # experiment named news_extraction_homework-<suffix>.
    results = evaluate(
        make_call_to_llm,
        data=dataset_name,
        evaluators=[perform_eval],
        experiment_prefix="news_extraction_homework",
        description="gpt-4o-mini structured extraction from news articles, scored by exact key match",
        metadata={"model": "gpt-4o-mini-2024-07-18", "temperature": 0},
        client=langsmith_client,
    )

    # Summarize the per-example scores locally as well
    scores = []
    for result in results:
        for eval_result in result["evaluation_results"]["results"]:
            if eval_result.score is not None:
                scores.append(eval_result.score)

    print(f"\nExperiment: {results.experiment_name}")
    if scores:
        print(f"Examples scored: {len(scores)}")
        print(f"Average score: {sum(scores) / len(scores):.2f}")
        print(f"Perfect extractions: {sum(1 for s in scores if s == 1.0)} / {len(scores)}")
    print("Open the experiment in LangSmith to inspect per-field mismatches.")
