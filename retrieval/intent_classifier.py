import os
import logging
from typing import Literal, Optional
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from langchain_groq import ChatGroq
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define intent categories as a Literal type for type-safety and enum-like behavior
IntentType = Literal["factual", "conceptual", "summarization", "ambiguous"]

# Pydantic model for structured output (ensures reliable parsing)
class IntentClassification(BaseModel):
    intent: IntentType = Field(description="The classified intent type")
    confidence: float = Field(description="Confidence score between 0.0 and 1.0")
    reasoning: str = Field(description="Brief explanation of the classification")

class IntentClassifier:
    """
    LLM-based query intent classifier using Groq's Llama 3.1 8B.
    Routes queries to adaptive retrieval strategies.
    Uses few-shot prompting for accuracy and structured output for reliability.
    """
    
    def __init__(
        self,
        model_name: str = "llama-3.1-8b-instant",
        temperature: float = 0.0,
        max_tokens: int = 150,
        api_key: Optional[str] = os.getenv("GROQ_API_KEY"),
    ):
        if not api_key:
            raise ValueError("GROQ_API_KEY not found in environment variables.")
        
        self.llm = ChatGroq(
            model=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=api_key,
        )
        
        # Parser for structured output
        self.parser = PydanticOutputParser(pydantic_object=IntentClassification)
        
        # Few-shot examples (curated from common RAG query patterns; add more for tuning)
        self.example_prompt = ChatPromptTemplate.from_messages(
            [
                ("human", "{query}"),
                ("ai", "{output}"),
            ]
        )
        
        self.few_shot_prompt = FewShotChatMessagePromptTemplate(
            example_prompt=self.example_prompt,
            examples=[
                {
                    "query": "What is the capital of France?",
                    "output": '{"intent": "factual", "confidence": 0.95, "reasoning": "Direct entity-based question seeking precise facts."}',
                },
                {
                    "query": "Explain how neural networks work.",
                    "output": '{"intent": "conceptual", "confidence": 0.90, "reasoning": "Seeks explanation of ideas and processes."}',
                },
                {
                    "query": "Summarize the history of AI.",
                    "output": '{"intent": "summarization", "confidence": 0.85, "reasoning": "Aggregates information over a broad topic."}',
                },
                {
                    "query": "What's the best way to learn programming?",
                    "output": '{"intent": "ambiguous", "confidence": 0.80, "reasoning": "Open-ended with multiple interpretations requiring clarification or expansion."}',
                },
            ],
        )
        
        # Full prompt template with instructions and parser format
        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", 
                 "You are an expert query intent classifier for a RAG system. "
                 "Classify the query into one of: factual (precise facts/entities), "
                 "conceptual (explanations/theories), summarization (overviews/aggregations), "
                 "ambiguous (unclear/multi-faceted). "
                 "Output ONLY the JSON as per the format instructions. "
                 "Provide confidence (0-1) and brief reasoning."
                ),
                self.few_shot_prompt,
                ("human", "{query}"),
                ("system", "{format_instructions}"),  # Inject parser instructions
            ]
        )
        
        # Chain: prompt -> LLM -> parser
        self.chain = self.prompt | self.llm | self.parser

    def classify(self, query: str) -> IntentClassification:
        """
        Classifies the query intent.
        
        Args:
            query (str): The user query to classify.
        
        Returns:
            IntentClassification: Structured result with intent, confidence, and reasoning.
        """
        try:
            format_instructions = self.parser.get_format_instructions()
            result = self.chain.invoke({"query": query, "format_instructions": format_instructions})
            logger.info(f"Classified query '{query[:50]}...' as {result.intent} (confidence: {result.confidence})")
            return result
        except Exception as e:
            logger.error(f"Classification failed for '{query}': {str(e)}")
            # Fallback: Default to 'ambiguous' with low confidence
            return IntentClassification(intent="ambiguous", confidence=0.5, reasoning="Fallback due to error.")

if __name__ == "__main__":
    # Quick test
    classifier = IntentClassifier()
    test_queries = [
        "Who invented the telephone?",
        "Compare quantum computing to classical computing.",
        "Give a summary of World War II.",
        "How do I fix my computer?"
    ]
    for q in test_queries:
        result = classifier.classify(q)
        print(f"Query: {q}\nIntent: {result.intent}\nConfidence: {result.confidence}\nReasoning: {result.reasoning}\n")