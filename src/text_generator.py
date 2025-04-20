from openai import OpenAI

class MistralGenerator:
    def __init__(self, model="mistral", base_url="http://localhost:1234/v1", api_key="lm-studio", prompt_template=None):
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model = model
        self.prompt_template = prompt_template or (
            "Context:\n{context}\n\n"
            "Question: {question}\n"
            "Answer:"
        )

    def generate_answer(self, question, context=""):
        prompt = self.prompt_template.format(context=context, question=question)
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=512,
        )
        return response.choices[0].message.content.strip()
