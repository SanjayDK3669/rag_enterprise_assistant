from langchain_google_genai import ChatGoogleGenerativeAI
from config.settings import GeminiConfig

def test_gemini() -> None:
    llm = ChatGoogleGenerativeAI(
        model = GeminiConfig.chat_model,
        google_api_key = GeminiConfig.api_key
    )
    
    response = llm.invoke("Say hello in one word")
    print(response.content)
    
if __name__ == "__main__":
    test_gemini()
