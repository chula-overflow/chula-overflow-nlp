import uvicorn

from typing import Union

from fastapi import FastAPI
from pydantic import BaseModel

from sentence_tokenizer import sentence_tokenize
from embed import embed_sentence
from measure import cosine_sim

app = FastAPI()

class TokenizeReqBody(BaseModel):
  paragraph: str


class EmbedReqBody(BaseModel):
  sentence: str

# class MeasureReqBody(BaseModel):
#   vector1: Union[float, float]
#   vector2: Union[float, float]

@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/tokenize")
def tokenize(body: TokenizeReqBody):
  sentences = sentence_tokenize(body.paragraph)
  return sentences

@app.post("/embed")
def embed(body: EmbedReqBody):
  embeded = embed_sentence(body.sentence)
  return {"vector": embeded}

@app.post("/measure")
def measure(body):
  similarity = cosine_sim(body.vector1, body.vector2)
  return similarity


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=3003)
