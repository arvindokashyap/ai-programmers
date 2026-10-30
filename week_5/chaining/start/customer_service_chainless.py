"""
Customer Service LLM Chainless Example

This module demonstrates how NOT to handle customer service responses by trying to do everything
in a single LLM call. This is meant to be compared with the chained approach to show why
breaking down complex tasks into smaller steps is better.

Requirements:
- Python 3.8+
- openai
"""

import os
import json
import time
from typing import Dict, List, Optional
import openai

# Rocket's TLS-inspecting firewall needs the Windows cert store
try:
    import truststore
    truststore.inject_into_ssl()
except ImportError:
    pass

# Configure your OpenAI API key (in a real app, use environment variables)
openai.api_key = os.environ.get("OPENAI_API_KEY")

MODEL = "gpt-4o-mini-2024-07-18"

class CustomerServiceChainless:
    """
    A class that demonstrates how NOT to handle customer service responses by trying to do
    everything in a single LLM call. This is meant to be compared with the chained approach.
    """

    def __init__(self):
        self.conversation_history: List[Dict[str, str]] = []
        # Track cost/latency so the two approaches can be compared on numbers
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.api_calls = 0

    def process_customer_question(self, customer_question: str) -> Dict[str, str]:
        """
        Process a customer question in a single LLM call (not recommended).

        Args:
            customer_question: The customer's question

        Returns:
            A dictionary containing all components of the response
        """
        # Try to do everything in one prompt (this is not the best approach!)
        response = openai.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": """You are a customer service assistant. For the given customer question:
                1. Create a short title (max 5 words)
                2. Analyze the question for:
                   - Main topic
                   - Customer's emotional state
                   - Key information needed
                   - Potential challenges
                3. Create a response strategy
                4. Write a final response

                Format your response as a JSON object with these keys:
                - title
                - analysis (as a nested object with the 4 components above)
                - strategy
                - response

                Be professional, empathetic, and thorough."""},
                {"role": "user", "content": customer_question}
            ]
        )
        self.api_calls += 1
        self.total_prompt_tokens += response.usage.prompt_tokens
        self.total_completion_tokens += response.usage.completion_tokens

        # Parse the response
        result = response.choices[0].message.content.strip()

        # The model often wraps JSON in a markdown code fence; strip it before parsing.
        # If parsing still fails we have no partial results to fall back on, which is
        # one of the weaknesses of the single-call approach.
        cleaned = result
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`")
            if cleaned.lower().startswith("json"):
                cleaned = cleaned[4:]
        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError:
            print("⚠️  Could not parse the model output as JSON; returning raw text.")
            parsed = {"raw": result}

        self.conversation_history.append({"question": customer_question, **parsed})
        return parsed

if __name__ == "__main__":
    # Initialize the chainless processor
    processor = CustomerServiceChainless()

    # Example customer question
    customer_question = """
    I've been trying to reset my password for the past hour, but I keep getting an error message
    saying 'Invalid email format'. I'm using the same email I've always used, and I'm getting
    really frustrated. Can you help me figure out what's going on?
    """

    # Process the question (all at once!)
    start = time.perf_counter()
    result = processor.process_customer_question(customer_question)
    elapsed = time.perf_counter() - start

    # Print the final response
    print("\n=== Final Response ===")
    if "response" in result:
        print(result["response"])
    else:
        print(result)

    print("\n=== Full Output ===")
    print(json.dumps(result, indent=2))

    print("\n=== Run Stats (chainless) ===")
    print(f"API calls:          {processor.api_calls}")
    print(f"Prompt tokens:      {processor.total_prompt_tokens}")
    print(f"Completion tokens:  {processor.total_completion_tokens}")
    print(f"Total tokens:       {processor.total_prompt_tokens + processor.total_completion_tokens}")
    print(f"Wall-clock time:    {elapsed:.2f}s")
