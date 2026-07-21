import ollama
from typing import Dict, Any, Generator

class BaseAgent:
    def __init__(self, model_name: str = "gemma4:26b"):
        self.model_name = model_name
        self.temperature = 0.0
        self.num_ctx = 32768
        self.num_predict = 2048
        self.keep_alive = "15m"

    def run_prompt(self, system_instruction: str, user_prompt: str, num_predict: int | None = None) -> str:
        """
        Runs a prompt against the Ollama model synchronously.
        """
        output_limit = num_predict or self.num_predict
        messages = [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": user_prompt}
        ]
        try:
            response = ollama.chat(
                model=self.model_name,
                messages=messages,
                stream=False,
                think=False,
                options={
                    "temperature": self.temperature,
                    "num_ctx": self.num_ctx,
                    "num_predict": output_limit
                },
                keep_alive=self.keep_alive,
            )
            
            # Extract content from response model/dict
            if hasattr(response, "message") and hasattr(response.message, "content"):
                return response.message.content or ""
            elif isinstance(response, dict):
                return response.get("message", {}).get("content", "")
            return str(response)
        except Exception as e:
            raise RuntimeError(f"Agent error during Ollama generation: {str(e)}")

    def run_prompt_stream(self, system_instruction: str, user_prompt: str, num_predict: int | None = None) -> Generator[str, None, None]:
        """
        Yields tokens chunk-by-chunk for streaming in the UI.
        """
        output_limit = num_predict or self.num_predict
        messages = [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": user_prompt}
        ]
        try:
            response_stream = ollama.chat(
                model=self.model_name,
                messages=messages,
                stream=True,
                think=False,
                options={
                    "temperature": self.temperature,
                    "num_ctx": self.num_ctx,
                    "num_predict": output_limit
                },
                keep_alive=self.keep_alive,
            )
            for chunk in response_stream:
                if 'message' in chunk and 'content' in chunk['message']:
                    yield chunk['message']['content']
        except Exception as e:
            yield f"\nAgent Error during streaming: {str(e)}"
