import ollama
from typing import Dict, Any, Generator

class BaseAgent:
    def __init__(self, model_name: str = "qwen3:8b"):
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

    def run_prompt_complete(self, system_instruction: str, user_prompt: str, num_predict: int | None = None, max_continuations: int = 3) -> str:
        """
        Like run_prompt, but detects when Ollama cut the response short (hit the
        num_predict ceiling) and automatically asks the model to continue until
        the file is actually complete, or max_continuations is reached.
        """
        output_limit = num_predict or self.num_predict
        messages = [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": user_prompt}
        ]
        full_content = ""

        for _ in range(max_continuations + 1):
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
            except Exception as e:
                raise RuntimeError(f"Agent error during Ollama generation: {str(e)}")

            if hasattr(response, "message"):
                chunk = response.message.content or ""
                done_reason = getattr(response, "done_reason", None)
            elif isinstance(response, dict):
                chunk = response.get("message", {}).get("content", "")
                done_reason = response.get("done_reason")
            else:
                chunk = str(response)
                done_reason = None

            import re
            cleaned_chunk = chunk.strip()
            cleaned_chunk = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", cleaned_chunk)
            cleaned_chunk = re.sub(r"\s*```$", "", cleaned_chunk)
            full_content += cleaned_chunk

            if done_reason != "length":
                break  # model finished on its own — nothing was cut off

            # Truncated: tell it to resume exactly where it stopped
            messages.append({"role": "assistant", "content": chunk})
            messages.append({"role": "user", "content": "Continue exactly where you left off — do not repeat any earlier lines and do not add commentary, just resume the code."})

        return full_content

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
