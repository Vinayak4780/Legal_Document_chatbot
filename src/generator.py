import os
from dotenv import load_dotenv
from groq import Groq
from langchain_core.language_models.llms import LLM
from typing import Any, List, Optional

# Load environment variables
load_dotenv()

class GroqGenerator(LLM):
    api_key: str = ""
    model_name: str = "llama3-8b-8192"
    client: Any = None
    
    def __init__(self, api_key=None, model_name="llama3-8b-8192"):
        super().__init__()
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("GROQ API key not found. Set GROQ_API_KEY environment variable or pass api_key parameter.")
        self.client = Groq(api_key=self.api_key)
        self.model_name = model_name
    @property
    def _llm_type(self):
        return "groq"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs: Any,
    ) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                max_tokens=kwargs.get("max_tokens", 1000),
                temperature=kwargs.get("temperature", 0.7),
                stream=False  # Keep non-streaming for LangChain compatibility
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error generating response: {str(e)}"
    
    def stream_call(
        self,
        prompt: str,
        max_tokens: int = 1000,
        temperature: float = 0.7,
        **kwargs: Any,
    ):
        """Stream tokens one by one from Groq API"""
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens,
                temperature=temperature,
                stream=True  # Enable streaming
            )
            
            for chunk in response:
                if chunk.choices[0].delta.content is not None:
                    yield chunk.choices[0].delta.content
                    
        except Exception as e:
            yield f"Error generating response: {str(e)}"
