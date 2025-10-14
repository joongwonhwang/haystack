"""Beauty product Q&A pipeline using Haystack for AMOREPACIFIC."""

from haystack import Pipeline
from haystack.components.retrievers import InMemoryBM25Retriever
from haystack.components.builders import PromptBuilder
from haystack.components.generators import OpenAIGenerator
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack import Document


BEAUTY_QA_TEMPLATE = """
You are a beauty expert at AMOREPACIFIC. Answer the question using only the provided context.
Context: {% for doc in documents %}{{ doc.content }} {% endfor %}
Question: {{ question }}
Answer:
"""


class BeautyQAPipeline:
    def __init__(self):
        self.store = InMemoryDocumentStore()
        self.pipeline = self._build_pipeline()

    def _build_pipeline(self) -> Pipeline:
        pipe = Pipeline()
        pipe.add_component("retriever", InMemoryBM25Retriever(document_store=self.store, top_k=3))
        pipe.add_component("prompt", PromptBuilder(template=BEAUTY_QA_TEMPLATE))
        pipe.add_component("llm", OpenAIGenerator(model="gpt-4o"))
        pipe.connect("retriever", "prompt.documents")
        pipe.connect("prompt", "llm")
        return pipe

    def index_products(self, products: list[dict]):
        docs = [
            Document(content=f"{p['name']}: {p['description']}", meta={"sku": p["sku"]})
            for p in products
        ]
        self.store.write_documents(docs)

    def ask(self, question: str) -> str:
        result = self.pipeline.run({"retriever": {"query": question}, "prompt": {"question": question}})
        return result["llm"]["replies"][0]


if __name__ == "__main__":
    qa = BeautyQAPipeline()
    qa.index_products([
        {"name": "Sulwhasoo Serum", "description": "Anti-aging ginseng serum for brightening.", "sku": "SW-001"},
        {"name": "Laneige Cream", "description": "Water-based moisturizer for all skin types.", "sku": "LG-002"},
    ])
    print(qa.ask("Which product is best for anti-aging?"))
