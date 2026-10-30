"""
Customer Service LLM Chain

This module demonstrates how to chain multiple LLM calls together to create
a comprehensive customer service response. The chain includes:
1. Creating a title for the conversation
2. Analyzing the customer's question
3. Generating a response strategy
4. Writing the final response

Requirements:
- Python 3.8+
- openai
"""

import os
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

class CustomerServiceChain:
    """
    A class that demonstrates chaining multiple LLM calls together to create
    a comprehensive customer service response.
    """

    def __init__(self):
        self.conversation_history: List[Dict[str, str]] = []
        # Track cost/latency so the two approaches can be compared on numbers
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.api_calls = 0
        self.step_timings: Dict[str, float] = {}

    def _call_llm(self, step: str, messages: List[Dict[str, str]]) -> str:
        """Single place for every LLM call so each step is timed and metered the same way."""
        start = time.perf_counter()
        response = openai.chat.completions.create(model=MODEL, messages=messages)
        self.step_timings[step] = time.perf_counter() - start
        self.api_calls += 1
        self.total_prompt_tokens += response.usage.prompt_tokens
        self.total_completion_tokens += response.usage.completion_tokens
        return response.choices[0].message.content.strip()

    def create_conversation_title(self, customer_question: str) -> str:
        """
        Create a concise title for the customer service conversation.

        Args:
            customer_question: The customer's initial question

        Returns:
            A title summarizing the conversation topic
        """
        return self._call_llm("title", [
            {"role": "system", "content": "You are a customer service assistant. Create a short, descriptive title (max 5 words) for this customer inquiry."},
            {"role": "user", "content": customer_question}
        ])

    def analyze_customer_question(self, customer_question: str) -> Dict[str, str]:
        """
        Analyze the customer's question to identify key components.

        Args:
            customer_question: The customer's question

        Returns:
            A dictionary containing analysis components
        """
        return self._call_llm("analysis", [
            {"role": "system", "content": """Analyze the customer's question and provide:
            1. Main topic
            2. Customer's emotional state
            3. Key information needed
            4. Potential challenges
            Format as a JSON object."""},
            {"role": "user", "content": customer_question}
        ])

    def generate_response_strategy(self, analysis: Dict[str, str]) -> str:
        """
        Generate a strategy for responding to the customer.

        Args:
            analysis: The analysis of the customer's question

        Returns:
            A strategy for crafting the response
        """
        return self._call_llm("strategy", [
            {"role": "system", "content": "Based on the analysis, create a response strategy that addresses the customer's needs while maintaining a professional and empathetic tone."},
            {"role": "user", "content": str(analysis)}
        ])

    def write_customer_response(self, customer_question: str, strategy: str) -> str:
        """
        Write the final response to the customer.

        Args:
            customer_question: The original customer question
            strategy: The response strategy

        Returns:
            The final response to send to the customer
        """
        return self._call_llm("response", [
            {"role": "system", "content": "Write a professional, helpful response to the customer's question. Be concise but thorough."},
            {"role": "user", "content": f"Question: {customer_question}\nStrategy: {strategy}"}
        ])

    def process_customer_question(self, customer_question: str) -> Dict[str, str]:
        """
        Process a customer question through the entire chain.

        Args:
            customer_question: The customer's question

        Returns:
            A dictionary containing all components of the response
        """
        # Step 1: Create a title
        title = self.create_conversation_title(customer_question)
        print(f"📝 Title: {title}")

        # Step 2: Analyze the question
        analysis = self.analyze_customer_question(customer_question)
        print(f"🔍 Analysis: {analysis}")

        # Step 3: Generate response strategy
        strategy = self.generate_response_strategy(analysis)
        print(f"🎯 Strategy: {strategy}")

        # Step 4: Write the response
        final_response = self.write_customer_response(customer_question, strategy)
        print(f"✉️ Response: {final_response}")

        # Store in conversation history
        self.conversation_history.append({
            "title": title,
            "question": customer_question,
            "analysis": analysis,
            "strategy": strategy,
            "response": final_response
        })

        return {
            "title": title,
            "analysis": analysis,
            "strategy": strategy,
            "response": final_response
        }


# Example usage
if __name__ == "__main__":
    # Initialize the chain
    chain = CustomerServiceChain()

    # Example customer question
    customer_question = """
    I've been trying to reset my password for the past hour, but I keep getting an error message
    saying 'Invalid email format'. I'm using the same email I've always used, and I'm getting
    really frustrated. Can you help me figure out what's going on?
    """

    # Process the question through the chain
    start = time.perf_counter()
    result = chain.process_customer_question(customer_question)
    elapsed = time.perf_counter() - start

    # Print the final response
    print("\n=== Final Response ===")
    print(result["response"])

    print("\n=== Run Stats (chained) ===")
    print(f"API calls:          {chain.api_calls}")
    print(f"Prompt tokens:      {chain.total_prompt_tokens}")
    print(f"Completion tokens:  {chain.total_completion_tokens}")
    print(f"Total tokens:       {chain.total_prompt_tokens + chain.total_completion_tokens}")
    print(f"Wall-clock time:    {elapsed:.2f}s")
    for step, seconds in chain.step_timings.items():
        print(f"  - {step:<9} {seconds:.2f}s")

    print("\n=== Why Chaining is Better ===")
    print("1. Each step can be optimized independently")
    print("2. Easier to debug and maintain")
    print("3. More reliable outputs for each component")
    print("4. Can reuse intermediate results")
    print("5. Better control over the process flow")
