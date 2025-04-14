from transformers import AutoProcessor, LlavaForConditionalGeneration
import json
import random
from PIL import Image
import os
import argparse

from openai import OpenAI

# Point to the local server
client = OpenAI(base_url="http://127.0.0.1:1235", api_key="lm-studio")

qasa_filtered_annotations_path = '../../../../Data/spiqa/test-B/SPIQA_testB.json'
with open(qasa_filtered_annotations_path, "r") as f:
    qasa_data = json.load(f)

_QASA_IMAGE_ROOT = "../../../../Data/spiqa/test-B/Images/SPIQA_testB_Images/SPIQA_testB_Images"#"../../../datasets/test-B/SPIQA_testB_Images"

_PROMPT = [
    """<image>\n Caption: <caption> Is the input image and caption helpful to answer the following question. Answer in one word - Yes or No. Question: <question>.\nASSISTANT:""",
    """<image>\n Caption: <caption> Please provide a brief answer to the following question after looking into the input image and caption. Question: <question>.\nASSISTANT:"""
]


def prepare_inputs(paper, question_idx):
    all_figures = list(paper['all_figures_tables'].keys())
    referred_figures = list(set(paper['referred_figures_tables'][question_idx]))
    answer = paper['composition'][question_idx]

    referred_figures_captions = []
    for figure in referred_figures:
        referred_figures_captions.append(paper['all_figures_tables'][figure])

    return answer, all_figures, referred_figures, referred_figures_captions

def infer_llava(qasa_data, args):
    _RESPONSE_ROOT = args.response_root
    os.makedirs(_RESPONSE_ROOT, exist_ok=True)

    for paper_id, paper in sorted(qasa_data.items(), key=lambda x: random.random()):
        output_path = os.path.join(_RESPONSE_ROOT, f"{paper_id}_response.json")
        if os.path.exists(output_path):
            continue

        response_paper = {}

        try:
            for question_idx, question in enumerate(paper['question']):
                answer, _, referred_figures, referred_figures_captions = prepare_inputs(paper, question_idx)
                answer_dict = {}

                for _idx, figure in enumerate(referred_figures):
                    caption = referred_figures_captions[_idx]
                    image_path = os.path.join(_QASA_IMAGE_ROOT, figure)
                    image = Image.open(image_path)
                    image = image.resize((args.image_resolution, args.image_resolution))

                    responses = []
                    for prompt_template in _PROMPT:
                        prompt = prompt_template.replace("<caption>", caption).replace("<question>", question)
                        result = client.text_to_image(prompt=prompt, image=image)
                        responses.append(result)

                    answer_dict[figure] = responses
                    print(answer_dict[figure])
                    print('-----------------')

                question_key = paper['question_key'][question_idx]
                response_paper[question_key] = {
                    'question': question,
                    'response': answer_dict,
                    'referred_figures_names': referred_figures,
                    'answer': answer
                }

        except Exception as e:
            print(f'Error: {e}')
            continue

        with open(output_path, 'w') as f:
            json.dump(response_paper, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate with LLaVA API on CPU.')
    parser.add_argument('--response_root', type=str, help='Response Root path.')
    parser.add_argument('--image_resolution', type=int, default=336, help='Image resolution.')
    args = parser.parse_args()

    infer_llava(qasa_data, args)    