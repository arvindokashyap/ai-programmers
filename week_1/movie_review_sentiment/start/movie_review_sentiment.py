# Corporate networks (e.g. Rocket) terminate TLS with an internal CA. Python's
# bundled certificate store doesn't include it, so verification fails. truststore
# defers to the Windows certificate store, which already trusts the internal CA.
try:
    import truststore

    truststore.inject_into_ssl()
except ImportError:
    pass

from openai import OpenAI

# Initialize OpenAI client
client = OpenAI()

def analyze_sentiment(review):
    """
    Analyze the sentiment of a movie review using structured output.
    Returns a dictionary with 'thought' and 'sentiment' keys.
    """
    prompt = f"""
    Analyze the sentiment of the following movie review and decide whether it is
    positive or negative overall.

    Review: {review}

    Respond in exactly this format, with no extra text:
    thought: your reasoning about what the reviewer liked and disliked
    sentiment: positive or negative
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini-2024-07-18",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7
    )

    content = response.choices[0].message.content

    # Parse the response, which is expected in the format:
    # thought: [analysis]
    # sentiment: [positive/negative]
    thought, sentiment = "", ""
    for line in content.strip().splitlines():
        line = line.strip()
        if line.lower().startswith("thought:"):
            thought = line.split(":", 1)[1].strip()
        elif line.lower().startswith("sentiment:"):
            sentiment = line.split(":", 1)[1].strip().strip('"\'.').lower()

    result = {
        "thought": thought,
        "sentiment": sentiment
    }

    return result

def main():
    # Test cases
    reviews = [
        "This film shouldn't work at all. It doesn't have much of a story and the whole dial up internet thing is incredibly dated. However Hanks and Ryan sell it beautifully.",
        "The movie was terrible. The acting was wooden, the plot made no sense, and I want my two hours back.",
        "An absolute masterpiece! The cinematography was stunning, the acting was superb, and the story kept me engaged from start to finish."
    ]

    # Test each review
    for i, review in enumerate(reviews, 1):
        result = analyze_sentiment(review)
        print(f"\nReview {i}:")
        print(f"Thought: {result['thought']}")
        print(f"Sentiment: {result['sentiment']}")

if __name__ == "__main__":
    main()
