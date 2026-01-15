import os
import logging
import time
import concurrent.futures
from typing import Literal, Optional, List
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from langchain_groq import ChatGroq
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import tiktoken

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define intent categories as a Literal type for type-safety and enum-like behavior
IntentType = Literal["factual", "conceptual", "summarization", "procedural", "comparison", "ambiguous"]

# Pydantic model for structured output (ensures reliable parsing)
class IntentClassification(BaseModel):
    intent: IntentType = Field(description="The classified intent type")
    confidence: float = Field(description="Confidence score between 0.0 and 1.0")
    reasoning: str = Field(description="Brief explanation of the classification")
    needs_expansion: bool = Field(default=False, description="Flag if query needs further processing (ambiguous or low confidence)")

class IntentClassifier:
    """
    LLM-based query intent classifier using Groq's Llama 3.1 8B.
    Routes queries to adaptive retrieval strategies.
    Uses few-shot prompting for accuracy and structured output for reliability.
    Supports batch processing with parallelism and rate-limit delays.
    """
    
    def __init__(
        self,
        model_name: str = "llama-3.1-8b-instant",
        temperature: float = 0.0,
        max_tokens: int = 150,
        api_key: Optional[str] = os.getenv("GROQ_API_KEY"),
        max_concurrent: int = 1,  # sequential to avoid rate limits
        base_delay: float = 2.0,  # Seconds; adjust based on Groq free-tier RPM (~30-60)
        max_retries: int = 3,
    ):
        if not api_key:
            raise ValueError("GROQ_API_KEY not found in environment variables.")
        
        self.llm = ChatGroq(
            model=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=api_key,
        )
        
        self.max_concurrent = max_concurrent
        self.base_delay = base_delay
        self.max_retries = max_retries
        self.token_estimator = tiktoken.get_encoding("cl100k_base")
        
        # Parser for structured output
        self.parser = PydanticOutputParser(pydantic_object=IntentClassification)
        
        # Few-shot examples (expanded with harder, multi-step, and edge cases for robustness)
        self.example_prompt = ChatPromptTemplate.from_messages(
            [
                ("human", "{query}"),
                ("ai", "{output}"),
            ]
        )
        
        self.few_shot_prompt = FewShotChatMessagePromptTemplate(
            example_prompt=self.example_prompt,
            examples=[
                # Basic cases
                {
                    "query": "What is the capital of France?",
                    "output": '{"intent": "factual", "confidence": 0.95, "reasoning": "Direct entity-based question seeking precise facts.", "needs_expansion": false}',
                },
                {
                    "query": "Explain how neural networks work.",
                    "output": '{"intent": "conceptual", "confidence": 0.90, "reasoning": "Seeks explanation of ideas and processes.", "needs_expansion": false}',
                },
                {
                    "query": "Summarize the history of AI.",
                    "output": '{"intent": "summarization", "confidence": 0.85, "reasoning": "Aggregates information over a broad topic.", "needs_expansion": false}',
                },
                {
                    "query": "Give me an overview of machine learning approaches.",
                    "output": '{"intent": "summarization", "confidence": 0.90, "reasoning": "Aggregates information over a broad topic.", "needs_expansion": false}',
                },
                {
                    "query": "How do I install Python on Windows?",
                    "output": '{"intent": "procedural", "confidence": 0.92, "reasoning": "Step-by-step instruction request.", "needs_expansion": false}',
                },
                {
                    "query": "Compare Python and Java for web development.",
                    "output": '{"intent": "comparison", "confidence": 0.95, "reasoning": "Explicit comparison between two entities.", "needs_expansion": false}',
                },
                {
                    "query": "What's the best way to learn programming?",
                    "output": '{"intent": "ambiguous", "confidence": 0.80, "reasoning": "Open-ended with multiple interpretations requiring clarification or expansion.", "needs_expansion": true}',
                },
                # Harder/edge cases: Multi-step, multi-class, ambiguous
                {
                    "query": "How does quantum entanglement work and how is it used in computing?",
                    "output": '{"intent": "conceptual", "confidence": 0.75, "reasoning": "Primarily explanatory but with multi-faceted elements; core is conceptual.", "needs_expansion": false}',
                },
                {
                    "query": "Tell me about Einstein's life, his theories, and compare relativity to quantum mechanics.",
                    "output": '{"intent": "ambiguous", "confidence": 0.65, "reasoning": "Multi-step (biography + explanations + comparison); unclear primary focus, needs decomposition.", "needs_expansion": true}',
                },
                {
                    "query": "Fix my code error and explain why it happened.",
                    "output": '{"intent": "procedural", "confidence": 0.70, "reasoning": "Step-by-step troubleshooting with explanation; borderline conceptual but procedural dominates.", "needs_expansion": false}',
                },
                {
                    "query": "What are the pros and cons of electric vs gas cars?",
                    "output": '{"intent": "comparison", "confidence": 0.88, "reasoning": "Clear comparative structure with balanced aspects.", "needs_expansion": false}',
                },
                {
                    "query": "Summarize climate change impacts and suggest solutions.",
                    "output": '{"intent": "ambiguous", "confidence": 0.60, "reasoning": "Mix of summarization and procedural (solutions); multi-class, requires expansion.", "needs_expansion": true}',
                },
            ],
        )
        
        # Improved prompt: Clearer definitions, handling for multi-step/ambiguous, emphasis on confidence thresholds
        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", 
                 "You are an expert query intent classifier for a RAG system. "
                 "Classify the query into EXACTLY ONE of: "
                 "factual (precise facts, entities, definitions, who/what/when/where), "
                 "conceptual (explanations, theories, how/why ideas work, relationships), "
                 "summarization (overviews, aggregations, timelines, broad histories), "
                 "procedural (step-by-step instructions, guides, how-to processes), "
                 "comparison (explicit contrasts between entities, pros/cons, differences/similarities), "
                 "ambiguous (unclear, multi-faceted, multi-step, or open-ended requiring clarification). "
                 "For multi-step or hybrid queries, choose the dominant intent or classify as ambiguous if no clear primary. "
                 "Set confidence <0.7 for uncertain/hybrid cases. "
                 "Set needs_expansion=true if ambiguous or confidence <0.7 for downstream query decomposition. "
                 "Output ONLY the JSON as per the format instructions. "
                 "Provide confidence (0-1 based on clarity/match) and brief reasoning."
                ),
                self.few_shot_prompt,
                ("human", "{query}"),
                ("system", "{format_instructions}"),  # Inject parser instructions
            ]
        )
        
        # Chain: prompt -> LLM -> parser
        self.chain = self.prompt | self.llm | self.parser

    def _estimate_tokens(self, query: str) -> int:
        """Approximate input tokens for logging/preempting limits."""
        format_instructions = self.parser.get_format_instructions()
        full_prompt = self.prompt.format(query=query, format_instructions=format_instructions)
        return len(self.token_estimator.encode(full_prompt))

    def classify(self, query: str) -> IntentClassification:
        """
        Classifies a single query intent with exponential backoff retries on 429 errors.
        
        Args:
            query (str): The user query to classify.
        
        Returns:
            IntentClassification: Structured result with intent, confidence, reasoning, and expansion flag.
        """
        token_est = self._estimate_tokens(query)
        logger.info(f"Estimated tokens for query '{query[:50]}...': {token_est}")
        
        retry_delay = self.base_delay
        for attempt in range(self.max_retries + 1):
            try:
                format_instructions = self.parser.get_format_instructions()
                result = self.chain.invoke({"query": query, "format_instructions": format_instructions})
                # Post-process: Enforce needs_expansion based on confidence/intent
                if result.intent == "ambiguous" or result.confidence < 0.7:
                    result.needs_expansion = True
                logger.info(f"Classified query '{query[:50]}...' as {result.intent} (confidence: {result.confidence}, needs_expansion: {result.needs_expansion})")
                return result
            except Exception as e:
                if "429" in str(e) and attempt < self.max_retries:
                    logger.warning(f"Rate limit hit (attempt {attempt+1}/{self.max_retries}). Retrying in {retry_delay}s...")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    logger.error(f"Classification failed for '{query}': {str(e)}")
                    # Fallback: Default to 'ambiguous' with expansion flag
                    return IntentClassification(intent="ambiguous", confidence=0.5, reasoning="Fallback due to error.", needs_expansion=True)
    def batch_classify(self, queries: List[str]) -> List[IntentClassification]:
        """
        Classifies multiple queries in parallel with concurrency limits and delays for rate limits.
        
        Args:
            queries (List[str]): List of queries to classify.
        
        Returns:
            List[IntentClassification]: Results in the same order as input queries.
        """
        results = [None] * len(queries)  # Preserve order
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_concurrent) as executor:
            future_to_idx = {executor.submit(self.classify, q): i for i,q in enumerate(queries)}
            for future in concurrent.futures.as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    results[idx] = future.result()
                except Exception as e:
                    logger.error(f"Batch classification failed for '{q}': {str(e)}")
                    results[idx] = IntentClassification(intent="ambiguous", confidence=0.5, reasoning="Batch fallback due to error.", needs_expansion=True)
                # Delay to respect rate limits
                time.sleep(self.base_delay)
        return results

if __name__ == "__main__":
    # Expanded quick test with more/harder examples
    classifier = IntentClassifier()
    test_queries = [
        "Who invented the telephone?",  # Factual
        "Compare quantum computing to classical computing.",  # Comparison
        "Give a summary of World War II.",  # Summarization
        "How do I fix my computer?",  # Procedural (potentially ambiguous)
        "Explain the theory of relativity and its applications in GPS.",  # Conceptual (multi-faceted)
        "What are the differences between supervised and unsupervised learning, and when to use each?",  # Comparison + procedural edge
        "Tell me everything about dinosaurs: their types, extinction, and modern theories.",  # Ambiguous (multi-step)
        "How to bake a cake from scratch while comparing oven vs microwave methods.",  # Procedural + comparison (hybrid, low confidence expected)
        "Summarize AI ethics debates and suggest best practices.",  # Summarization + procedural (ambiguous)
        "Is climate change real? Provide facts, explanations, and comparisons to past events."  # Ambiguous (multi-class)
    ]
    # Test single classify
    for q in test_queries[:5]:  # First 5 as single
        result = classifier.classify(q)
        print(f"Query: {q}\nIntent: {result.intent}\nConfidence: {result.confidence}\nReasoning: {result.reasoning}\nNeeds Expansion: {result.needs_expansion}\n")
    
    # Test batch classify on remaining
    batch_results = classifier.batch_classify(test_queries[5:])
    for q, res in zip(test_queries[5:], batch_results):
        print(f"Batch Query: {q}\nIntent: {res.intent}\nConfidence: {res.confidence}\nReasoning: {res.reasoning}\nNeeds Expansion: {res.needs_expansion}\n")