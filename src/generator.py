from openai import OpenAI
from image_generator import LlavaGenerator
from icecream import ic

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

class MultimodalAnswerPipeline:
    def __init__(self, mistral: MistralGenerator, llava: LlavaGenerator):
        self.mistral = mistral
        self.llava = llava

    def needs_visual_analysis(self, caption, question):
        detection_prompt = f"""
You are a helpful scientific assistant. Based on the caption and question below, determine if answering the question requires analyzing the visual (figure, table, graph, etc.) or if it can be answered from text alone.

Caption: "{caption}"
Question: "{question}"

Instructions:
- If the caption alone is enough to fully answer the question, reply: "No visual needed."
- If more visual context is needed, reply: "Visual needed."

Only respond with exactly one of the two phrases.
"""
        response = ic(self.mistral.generate_answer(question=detection_prompt, context=""))
        return "visual needed" in response.lower()

    def synthesize_answer(self, question, caption, vlm_answer):
        synthesis_prompt = f"""
You are a scientific assistant. Based on the information below, synthesize a final complete answer:

Original Question: {question}
Caption: {caption}
Visual Model's Answer: {vlm_answer}

Instructions:
- Write a complete, coherent final answer.
- Integrate information from both caption and visual model where appropriate.
"""
        return ic(self.mistral.generate_answer(question=synthesis_prompt, context=""))
    def generate_visual_subquestions(self, caption, question):
        subq_prompt = f"""
    You are a scientific assistant. Based on the caption and question below, generate the specific visual-related subquestions that would help in answering.

    Caption: "{caption}"
    Question: "{question}"

    List the subquestions clearly. If none needed, say 'None'.
    """
        response = self.mistral.generate_answer(question=subq_prompt, context="")
        return response

    def answer_question(self, question, caption, image_path):
        if self.needs_visual_analysis(caption, question):
            subquestions_text = self.generate_visual_subquestions(caption, question)
            ic(subquestions_text)
            questions = f"""main question:{question}, subquestions:{subquestions_text}"""
            if "none" in subquestions_text.lower():
                vlm_answer = self.llava.generate_answer(image_path=image_path, question=question, caption=caption)
            else:
                # Ask VLM the generated subquestions instead of original question
                vlm_answer = self.llava.generate_answer(image_path=image_path, question=questions, caption=caption)
            ic(vlm_answer)
            final_answer = self.synthesize_answer(question=question, caption=caption, vlm_answer=vlm_answer)
        else:
            final_answer = self.mistral.generate_answer(question=question, context=caption)
            
        return final_answer

