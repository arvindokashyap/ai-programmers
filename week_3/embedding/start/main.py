from openai import OpenAI
import faiss
import numpy

# Rocket's TLS-inspecting firewall needs the Windows cert store
try:
    import truststore
    truststore.inject_into_ssl()
except ImportError:
    pass

# Example documents about movies
texts = [
    "The Godfather is a classic mafia crime drama",
    "Inception explores dreams within dreams",
    "The Shawshank Redemption is a story about hope and friendship",
]

client = OpenAI()

def get_embedding(text):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding

embeddings = []
for text in texts:
    embedding = get_embedding(text)
    embeddings.append(embedding)


dimension = len(embeddings[0])
index = faiss.IndexFlatL2(dimension)

index.add(numpy.array(embeddings, dtype='float32'))

query = 'Tell me about a prison movie'

query_embedding = get_embedding(query)
distances, indicies = index.search(numpy.array([query_embedding], dtype='float32'), 3)

print("Best Matches with Distance (lower is better)")
for i in range(3):
    print(f"Match {i+1}, Distance: {distances[0][i]:.4f}")
    print(texts[indicies[0][i]]) 