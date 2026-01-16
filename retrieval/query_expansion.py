import os
import logging
import time
import concurrent.futures
from typing import List, Optional
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from langchain_groq import ChatGroq
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from intent_classifier import IntentType, IntentClassification  # Import from sibling module

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Updated Pydantic model: Separate HyDE field for independent embedding
class ExpandedQuery(BaseModel):
    original_query: str = Field(description="The original user query")
    expanded_queries: List[str] = Field(description="List of rephrased (Multi-Query) or decomposed (Decomposition) queries (exactly max_expansions)")
    hyde_hypothetical: Optional[str] = Field(default=None, description="Separate hypothetical answer for HyDE (200-300 words if generated)")
    reasoning: str = Field(description="CoT reasoning for expansion choices")

class QueryExpander:
    """
    Enhanced LLM-based query expander using Groq's Llama 4 Scout.
    Separates HyDE from expansions to avoid vector skew.
    Distinguishes Multi-Query (rephrasing) vs. Decomposition (sequential sub-steps).
    Uses CoT for reasoning; stricter constraints for output control.
    Adaptive for ambiguous: HyDE for short/vague, Decomposition for multi-faceted.
    Supports batch with concurrency limits.
    """
    
    def __init__(
        self,
        model_name: str = "meta-llama/llama-4-scout-17b-16e-instruct",
        temperature: float = 0.5,  # Increased for creativity, balanced by constraints
        max_tokens: int = 1024,  # Room for HyDE/CoT
        api_key: Optional[str] = os.getenv("GROQ_API_KEY"),
        max_concurrent: int = 1,
        base_delay: float = 2.0,
        max_retries: int = 3,
        max_expansions: int = 5,
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
        self.max_expansions = max_expansions
        
        # Parser
        self.parser = PydanticOutputParser(pydantic_object=ExpandedQuery)
        
        # Updated few-shot: Distinguish Multi-Query vs. Decomposition
        self.example_prompt = ChatPromptTemplate.from_messages(
            [
                ("human", "{query} | Intent: {intent}"),
                ("ai", "{output}"),
            ]
        )
        
        self.few_shot_prompt = FewShotChatMessagePromptTemplate(
            example_prompt=self.example_prompt,
            examples = [
                # 1. FACTUAL: Multi-Query (Focus on Lexical Diversity)
                {
                    "query": "Capital of France?",
                    "intent": "factual",
                    "output": """{
                        "original_query": "Capital of France?",
                        "expanded_queries": ["What is the capital city of France?", "France's main administrative city?", "Official capital of the French Republic?"],
                        "hyde_hypothetical": null,
                        "reasoning": "Step1: Intent is factual. Step2: Strategy is Multi-Query to catch different keyword variations. Step3: No HyDE required for specific lookup."
                    }"""
                },
                # 2. PROCEDURAL (Complex): Decomposition (Focus on Sequential Steps)
                {
                    "query": "How do I migrate from AWS to GCP?",
                    "intent": "procedural",
                    "output": """{
                        "original_query": "How do I migrate from AWS to GCP?",
                        "expanded_queries": [
                            "Assessment and planning for AWS to GCP cloud migration",
                            "How to map AWS IAM roles to GCP IAM policies",
                            "Data migration tools for moving S3 buckets to Google Cloud Storage",
                            "Strategies for migrating AWS RDS instances to GCP Cloud SQL"
                        ],
                        "hyde_hypothetical": null,
                        "reasoning": "Step1: Intent is complex procedural. Step2: Strategy is Decomposition into logical infrastructure phases. Step3: Targeted queries allow better coverage of specific migration tasks."
                    }"""
                },
                # 3. CONCEPTUAL/SUMMARIZATION: HyDE + Decomposition (Focus on Semantic Depth)
                {
                    "query": "The impact of the Industrial Revolution on modern society.",
                    "intent": "conceptual",
                    "output": """{
                        "original_query": "The impact of the Industrial Revolution on modern society.",
                        "expanded_queries": [
                            "Long-term economic effects of the Industrial Revolution",
                            "Social changes and urbanization resulting from the 19th-century industrial shift",
                            "How industrialization shaped modern labor laws and education systems"
                        ],
                        "hyde_hypothetical": "The Industrial Revolution marked a fundamental shift from agrarian economies to industrialized urban centers. Its legacy includes the rise of the middle class, the globalization of trade, and the transition to fossil-fuel-based energy. Modern technological advancement is a direct evolution of this period, influencing everything from urban infrastructure to current labor rights and global economic policy.",
                        "reasoning": "Step1: Intent is broad and conceptual. Step2: Strategy is HyDE for semantic mapping plus Decomposition into socio-economic sub-topics. Step3: HyDE provides a rich context vector for retrieval."
                    }"""
                },
                # 4. AMBIGUOUS (Short): HyDE-Heavy (Focus on Intent Clarification)
                {
                    "query": "Best AI learning path?",
                    "intent": "ambiguous",
                    "output": """{
                        "original_query": "Best AI learning path?",
                        "expanded_queries": ["What are the fundamental steps to learn Artificial Intelligence?", "AI roadmap for beginners vs advanced learners"],
                        "hyde_hypothetical": "A comprehensive AI learning path begins with core mathematics like linear algebra and statistics, followed by mastering Python. Students then typically progress to supervised machine learning, deep learning architectures, and eventually specialized fields like Natural Language Processing (NLP) or Computer Vision (CV). Projects and hands-on coding are essential components.",
                        "reasoning": "Step1: Query is short and vague. Step2: Strategy is HyDE-Heavy to generate a 'proto-answer' that bridges the gap between the user's short query and dense technical documentation. Step3: Minimal rephrasing to maintain scope."
                    }"""
                },
                # 5. COMPARISON: Dimensional Decomposition (Focus on Side-by-Side Analysis)
                {
                    "query": "Llama 3 vs GPT-4o for coding",
                    "intent": "comparison",
                    "output": """{
                        "original_query": "Llama 3 vs GPT-4o for coding",
                        "expanded_queries": [
                            "Llama 3 performance on HumanEval and MBPP coding benchmarks",
                            "GPT-4o programming capabilities and software engineering efficiency",
                            "Comparison of Llama 3 and GPT-4o in Python and Javascript generation"
                        ],
                        "hyde_hypothetical": null,
                        "reasoning": "Step1: Intent is comparison across specific models. Step2: Strategy is Decomposition by dimension (benchmark scores vs. language support). Step3: Retrieval targets data points for both models simultaneously."
                    }"""
                }
            ]
        )
        
        # prompt: CoT, distinctions, constraints, adaptive ambiguous
        self.prompt = ChatPromptTemplate.from_messages(
        [
            ("system", 
            "You are a query expansion expert. Your response must be **ONLY** a valid JSON object\n"

            "Use Chain-of-Thought (CoT) in reasoning:\n"
            "Step1: Analyze intent/query length/complexity.\n"
            "Step2: Choose primary strategy.\n"
            "Step3: Generate.\n"

            "Strategies (choose ONE primary):\n"
            "- factual: Multi-Query → parallel rephrasings + synonyms to overcome keyword mismatch.\n"
            "- procedural: Decomposition → break into sequential, ordered sub-steps/phases so every stage is covered.\n"
            "- comparison: Decomposition → break into aspects/criteria/entities to compare (e.g. separate queries per item or per dimension).\n"
            "- conceptual: HyDE (hypothetical detailed answer, 200-300 words) + light decomposition if multi-faceted.\n"
            "- summarization: HyDE (broad hypothetical overview) or light decomposition.\n"
            "- ambiguous: Adaptive via CoT → short/vague → prioritize HyDE; long/multi-faceted → Decomposition.\n"
            
            "Constraints:\n"
            "- Generate EXACTLY {max_expansions} expanded_queries.\n"
            "- Maintain technical terminology; do not over-simplify or paraphrase concepts incorrectly.\n"
            "- HyDE is a separate field (only generate when strategy includes it).\n"
            "- Output ONLY a valid JSON object as per format instructions. No extra text."
            ),
            self.few_shot_prompt,
            ("human", "{query} | Intent: {intent}"),
            ("system", "{format_instructions}"),
        ])
        
        # Chain
        self.chain = self.prompt | self.llm | self.parser

    def expand(self, query: str, intent_classification: IntentClassification) -> ExpandedQuery:
        """
        Expands a single query with checks/skips as before.
        """
        intent = intent_classification.intent
        if not intent_classification.needs_expansion and intent not in ["ambiguous", "conceptual", "summarization", "procedural", "comparison"]:
            return ExpandedQuery(
                original_query=query,
                expanded_queries=[query] * self.max_expansions if self.max_expansions > 1 else [query],
                hyde_hypothetical=None,
                reasoning="No expansion needed; replicated original for consistency."
            )
        
        retry_delay = self.base_delay
        for attempt in range(self.max_retries + 1):
            try:
                format_instructions = self.parser.get_format_instructions()
                result = self.chain.invoke({
                    "query": query,
                    "intent": intent,
                    "max_expansions": self.max_expansions,
                    "format_instructions": format_instructions
                })
                # Post-process: Enforce exact count (pad/truncate if LLM drifts)
                if len(result.expanded_queries) != self.max_expansions:
                    result.expanded_queries = (result.expanded_queries[:self.max_expansions] + [query] * (self.max_expansions - len(result.expanded_queries)))[:self.max_expansions]
                logger.info(f"Expanded '{query[:50]}...' for '{intent}' (reasoning: {result.reasoning})")
                return result
            except Exception as e:
                if "429" in str(e) and attempt < self.max_retries:
                    logger.warning(f"Rate limit (attempt {attempt+1}/{self.max_retries}). Retrying in {retry_delay}s...")
                    time.sleep(retry_delay)
                    retry_delay *= 2
                else:
                    logger.error(f"Failed for '{query}': {str(e)}")
                    return ExpandedQuery(
                        original_query=query,
                        expanded_queries=[query] * self.max_expansions,
                        hyde_hypothetical=None,
                        reasoning="Fallback due to error."
                    )

    def batch_expand(self, queries: List[str], intent_classifications: List[IntentClassification]) -> List[ExpandedQuery]:
        """Batch as before, with updated expand."""
        results = [None] * len(queries)
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_concurrent) as executor:
            future_to_idx = {executor.submit(self.expand, q, ic): i for i, (q, ic) in enumerate(zip(queries, intent_classifications))}
            for future in concurrent.futures.as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    results[idx] = future.result()
                except Exception as e:
                    logger.error(f"Batch failed for '{queries[idx]}': {str(e)}")
                    results[idx] = ExpandedQuery(
                        original_query=queries[idx],
                        expanded_queries=[queries[idx]] * self.max_expansions,
                        hyde_hypothetical=None,
                        reasoning="Batch fallback."
                    )
                time.sleep(self.base_delay)
        return results

if __name__ == "__main__":
    # Updated test: Use classifier
    from intent_classifier import IntentClassifier
    classifier = IntentClassifier()
    expander = QueryExpander(max_expansions=3)  # Test with smaller max
    
    test_queries = [
        "Capital of India?",  # Factual: Multi-Query
        "Tell me about Newton's life and theories.",  # Ambiguous multi-faceted: Decomposition + HyDE
        "Best AI learning path?",  # Ambiguous short: HyDE + minimal
        "Install Python on Windows."  # Procedural: Multi-Query
    ]
    
    classifications = classifier.batch_classify(test_queries)
    expansions = expander.batch_expand(test_queries, classifications)
    
    for q, exp in zip(test_queries, expansions):
        print(f"Query: {q}")
        print(f"Expanded: {exp.expanded_queries}")
        if exp.hyde_hypothetical:
            print(f"HyDE: {exp.hyde_hypothetical[:100]}...")
        print(f"Reasoning: {exp.reasoning}\n")