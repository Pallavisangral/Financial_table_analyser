r""" server.openai_api module """

# importing standard modules ==================================================
import os

# importing third-party modules ===============================================
import traceback
import openai
from openai import OpenAI

# importing custom modules ====================================================
from logger import LoggingHandle
from model import Responder

# class definitions  ==========================================================
# gpt-3.5-turbo- Total 16K tokens


class Openai_API:
    def __init__(self):
        # openai_key = os.environ.get("OPENAI_KEY")
        self.client = OpenAI(api_key="add_key")
        
    def generate_zero_shot_text(self, message):
        pass

    def generate_few_shot_text(
        self,
        messages,
        model="gpt-3.5-turbo",
        max_tokens=700,
        temperature=0.4,
        top_p: float = None,
        frequency_penalty: float = None,
        seed: int = None,
    ):
        try:
            chat_dict = {
                "messages": messages,
                "model": model,
                "max_tokens": max_tokens,
                "n": 1,
                "temperature": temperature,
            }
            if top_p:
                chat_dict["top_p"] = top_p
            if frequency_penalty:
                chat_dict["frequency_penalty"] = frequency_penalty
            if seed:
                chat_dict["seed"] = seed

            response = self.client.chat.completions.create(**chat_dict)
            generated_text = response.choices[0].message.content
            return Responder(status="success", message=generated_text)

        except openai.APIError as e:
            LoggingHandle().write_log(traceback.format_exc())
            # error_message = e._message or 'Invalid request'
            return Responder(status="error", message=str(e))
        except openai.APIConnectionError as e:

            LoggingHandle().write_log(traceback.format_exc())
            return Responder(status="error", message=str(e))
        except openai.RateLimitError as e:

            LoggingHandle().write_log(traceback.format_exc())
            return Responder(status="error", message=str(e))
        except Exception as e:
            LoggingHandle().write_log(traceback.format_exc())
            return Responder(status="error", message=str(e))

    def generate_few_shot_text_with_n(self,
        messages,
        model="gpt-3.5-turbo",
        max_tokens=700,
        n=1,
        temperature=0.4,
        top_p: float = None,
        frequency_penalty: float = None,
        seed:int=None):
        try:
            chat_dict = {
                "messages": messages,
                "model": model,
                "max_tokens": max_tokens,
                "n": n,
                "temperature": temperature,
            }
            if top_p:
                chat_dict["top_p"] = top_p
            if frequency_penalty:
                chat_dict["frequency_penalty"] = frequency_penalty
            if seed:
                chat_dict["seed"] = seed
            response = self.client.chat.completions.create(**chat_dict)
            # generated_text = response.choices[0].message.content
            generated_text = ""
            for i, result in enumerate(response.choices):
                generated_text += f"{result.message.content}\n"
            return Responder(status="success", message=generated_text)
        except openai.APIError as e:
            LoggingHandle().write_log(traceback.format_exc())
            # error_message = e._message or 'Invalid request'
            return Responder(status="error", message=str(e))
        except openai.APIConnectionError as e:
            LoggingHandle().write_log(traceback.format_exc())
            return Responder(status="error", message=str(e))
        except openai.RateLimitError as e:
            LoggingHandle().write_log(traceback.format_exc())
            return Responder(status="error", message=str(e))
        except Exception as e:
            LoggingHandle().write_log(traceback.format_exc())
            return Responder(status="error", message=str(e))