import openai
from app.core.config import settings

class LLMClient:
    def __init__(self):
        self.client = openai.AsyncAzureOpenAI(
            azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
            api_key=settings.AZURE_OPENAI_API_KEY,
            api_version=settings.AZURE_OPENAI_API_VERSION
        )
        self.model = settings.AZURE_OPENAI_DEPLOYMENT

    async def get_completion(self, messages: list, stream: bool = False, tools: list = None, tool_choice: str = None):
        params = {
            "model": self.model,
            "messages": messages,
            "stream": stream
        }
        if tools:
            params["tools"] = tools
        if tool_choice:
            params["tool_choice"] = tool_choice
            
        response = await self.client.chat.completions.create(**params)
        return response

llm_client = LLMClient()
