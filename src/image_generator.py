import base64
import os
from openai import OpenAI
from icecream import ic

def foo(i):
    return i + 333

ic(foo(123))

class LlavaGenerator:
    def __init__(self, model="llava-v1.5-7b@q4_k_m", base_url="http://localhost:1234/v1", api_key="lm-studio", prompt_template=None):
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model = model
        self.prompt_template = prompt_template or (
            "<image>\n\nPlease answer the following question based only on the image .\nQuestion: <question>\nAnswer:"
        )

    def generate_answer(self, image_path, question,caption=""):
        prompt = self.prompt_template.replace("<question>", question)

        with open(image_path, "rb") as f:
            image_base64 = base64.b64encode(f.read()).decode("utf-8")

        messages = [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}},
                {"type": "text", "text": prompt}
            ]
        }]

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.7,
            max_tokens=512
        )
        result =response.choices[0].message.content.strip()
        ic(result)
        return result

        # return response.choices[0].message.content.strip()
