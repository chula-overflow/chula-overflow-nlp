from sentence_transformers import SentenceTransformer

from measure import cosine_sim
model = SentenceTransformer('bert-base-nli-mean-tokens', use_auth_token='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')

def embed_sentence(sentence):
    """
    Input - (String) sentence to be embedded
    out - (numpy.ndarray) shape of (768, ) embedded vector of a sentence
    """

    encoded = [float(n) for n in model.encode(sentence)]
    return encoded

sim = cosine_sim(embed_sentence("which statement is true about electron and hole mobility in semiconductors?"), embed_sentence('which is true about electron and hole mobility in semiconductors?'))

print(sim)
