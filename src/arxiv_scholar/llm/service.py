import os
import json
import logging
from typing import List, AsyncGenerator, Any
from openai import AsyncOpenAI

logger = logging.getLogger(__name__)

class LLMService:
    def __init__(self, api_key: str = None, base_url: str = None, model: str = None):
        self.api_key = api_key or os.environ.get("LLM_API_KEY") or os.environ.get("GEMINI_API_KEY", "")
        self.base_url = base_url or os.environ.get("LLM_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/")
        self.model = model or os.environ.get("LLM_MODEL", "gemma-4-31b-it")
        
        if self.api_key:
            self.client = AsyncOpenAI(
                base_url=self.base_url,
                api_key=self.api_key
            )
        else:
            self.client = None
            logger.warning("No API Key provided for LLMService. API calls will fail or fall back.")

    async def generate_hyde_abstract(self, query: str) -> str:
        if not self.client:
            raise ValueError("LLM client not initialized.")
            
        prompt = f"""
        Please write a brief academic abstract that directly answers the following query. 
        Do not use conversational filler, just write the abstract.
        Query: {query}
        """
        logger.info(f"Requesting HyDE abstract from LLM ({self.model}) for query: '{query}'")
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        return response.choices[0].message.content

    async def decompose_query(self, query: str) -> Any:
        if not self.client:
            logger.warning("No LLM client configured for DECOMPOSE. Falling back.")
            return {"sub_queries": [query]}
            
        prompt = f"""
        Decompose this query into independent, fully contextualized sub-queries. 
        Each sub-query must be able to stand completely on its own for a search engine.
        Also, extract any explicit metadata constraints (e.g., year published) into a structured filters object.
        
        For example: "Accuracy of BERT vs GPT-3 published after 2022" -> 
        sub_queries: ["What is the accuracy of BERT?", "What is the accuracy of GPT-3?"]
        filters: {{"year": {{"operator": ">=", "value": 2022}}}}
        
        Valid operators: ">=", ">", "<=", "<", "=="
        
        Query: {query}
        
        Return ONLY valid JSON in this exact format:
        {{
            "sub_queries": ["sub query 1", "sub query 2"],
            "filters": {{
                "year": {{
                    "operator": ">=",
                    "value": 2022
                }}
            }}
        }}
        Omit the "filters" key entirely if there are no metadata constraints.
        """
        logger.info(f"Requesting query decomposition from LLM ({self.model}) for query: '{query}'")
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.1
        )
        content = response.choices[0].message.content.strip()
        
        # Safely strip markdown formatting if the model included it despite the json_object constraint
        if content.startswith("```json"):
            content = content[7:]
        if content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]
        content = content.strip()
        
        try:
            data = json.loads(content)
            return data
        except Exception as e:
            logger.error(f"Failed to parse LLM JSON for decomposition: {e} | Content: {content}")
            return {"sub_queries": [query]}

    async def stream_synthesis(self, query: str, context: str) -> AsyncGenerator[str, None]:
        if not self.client:
            raise ValueError("LLM client not initialized.")
            
        prompt = f"""You are an academic research assistant. 
Answer the user's query comprehensively based ONLY on the provided context chunks. 
If the answer is not in the context, state that clearly. Cite your sources where applicable using the provided Source URL (e.g. [https://arxiv.org/abs/2010.05432]).

CRITICAL: Return ONLY the final answer. Do NOT use conversational filler (e.g. "Based on the provided context..."). Start your answer immediately.

Context:
{context}

Query: {query}
"""
        logger.info(f"Starting LLM synthesis stream ({self.model}) for query: '{query}'")
        stream = await self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            stream=True
        )
        
        in_thought = False
        buffer = ""
        
        async for chunk in stream:
            content = chunk.choices[0].delta.content
            if not content:
                continue
                
            buffer += content
            
            # Simple state machine to hide <thought> tags from the UI
            while True:
                if not in_thought:
                    if "<thought>" in buffer:
                        idx = buffer.find("<thought>")
                        if idx > 0:
                            yield buffer[:idx]
                        buffer = buffer[idx + len("<thought>"):]
                        in_thought = True
                    else:
                        split_idx = buffer.find("<")
                        if split_idx == -1:
                            yield buffer
                            buffer = ""
                            break
                        else:
                            if split_idx > 0:
                                yield buffer[:split_idx]
                            buffer = buffer[split_idx:]
                            if len(buffer) < len("<thought>"):
                                if "<thought>".startswith(buffer):
                                    break # Wait for more chunks to resolve the tag
                                else:
                                    yield buffer[0]
                                    buffer = buffer[1:]
                            else:
                                yield buffer[0]
                                buffer = buffer[1:]
                else:
                    if "</thought>" in buffer:
                        idx = buffer.find("</thought>")
                        buffer = buffer[idx + len("</thought>"):]
                        in_thought = False
                    else:
                        split_idx = buffer.find("<")
                        if split_idx == -1:
                            buffer = ""
                            break
                        else:
                            buffer = buffer[split_idx:]
                            if len(buffer) < len("</thought>"):
                                if "</thought>".startswith(buffer):
                                    break
                                else:
                                    buffer = buffer[1:]
                            else:
                                buffer = buffer[1:]
                                
        if buffer and not in_thought:
            yield buffer
