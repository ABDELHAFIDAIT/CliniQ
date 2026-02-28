import re
from deepeval.models.base_model import DeepEvalBaseLLM
from langchain_ollama import ChatOllama
from app.core.config import settings


class EvalService(DeepEvalBaseLLM):
    def __init__(self):
        self.model = ChatOllama(
            model=settings.EVAL_LLM_MODEL,
            base_url=settings.OLLAMA_BASE_URL,
            temperature=0,
            format="json" 
        )
    
    
    def load_model(self):
        return self.model
    
    
    def clean_json_output(self, output: str) -> str:
        match = re.search(r'(\{.*\}|\[.*\])', output, re.DOTALL)
        if match:
            return match.group(1)
        return output
    
    
    def generate(self, prompt: str) -> str:
        res = self.model.invoke(prompt)
        return self.clean_json_output(res.content)


    async def a_generate(self, prompt: str) -> str:
        res = await self.model.ainvoke(prompt)
        return self.clean_json_output(res.content)


    def get_model_name(self):
        return f"Ollama {settings.EVAL_LLM_MODEL}"
    
    
eval_service = EvalService()