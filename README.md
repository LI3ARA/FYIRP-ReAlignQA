# Multimodal RAG & VQA Research Repository - ReAlignQA

This repository explores usage of multimodal data(Images and Text) for question answering based on given text and image data. 

---

## Project Structure

```bash
.
├── Notebooks/            # Experiments organized by date or topic
│   ├── 24/               # Earlier experiments (LangChain-based RAG)
│   └── 25/
│       ├── LayoutLM+RAG/     # Layout-aware document QA
│       ├── VQA_RAG/          # LLaVA-based VQA pipelines <--- Main experiments
│       └── ...               # Other vision-based extraction tools
├── src/                 # Core Python modules
├── Utils/               # Helper scripts, e.g., notebook outlining
├── notebooks/en/        # Cleaned examples for multimodal RAG
├── docs/                # Project documentation / UI
├── requirements.txt     # Python dependencies
└── .gitignore
```
---

## Installation
Using a conda environment is reccommended
```
conda create -n multimodalrag python=3.11
conda activate multimodalrag
pip install -r requirements.txt
```
Also configure your environment variables:
- Create a `.env` file in the root directory.
- Add any API keys for the models called under OpenAI wrapper.

`.env` example
```
REMOTE_URL=your-remote\local-llm-url
LOCAL_URL=your-remote\local-lmm-url
API_KEY=your-api-key
```

## Evaluation
- Metrics Functions: see `src/eval_metrics_utils.py`
- Evaluation notebooks:
  - `run_text_eval.ipynb`
  - `run_vision_only_eval.ipynb`
- Outputs are saved to structured CSVs in `../Eval_outputs/SPIQA/vision_only/`

## Datasets Used
| Dataset        | Purpose                                                 |
| -------------- | ------------------------------------------------------- |
| **SPIQA**      | Visual QA dataset from structured documents             |
| **PDF-VQA**    | PDF-based VQA task for layout-sensitive models          |
| **VisDoMRAG** | Extension for SPIQA dataset |


## Models Used
| Model        | Use                                       |
| ------------ | ----------------------------------------- |
| `LLaVA`      | Vision-language inference and fine-tuning |
| `LayoutLMv3` | OCR-aware document representation         |
| `Mistral`    | Text-only RAG baseline                    |
| `LLaVA and Mistral` | For RealignQA |

## UI & Deployment
- Gradio demo: `src/multimodal_gradio_UI.ipynb`
- Colab scripts for LLaVA fine tuning and evaluation: `VQA_RAG/Googl_colab/soft_prompting/`
  
## Cite the Work
[![DOI](https://zenodo.org/badge/884740883.svg)](https://doi.org/10.5281/zenodo.15471173)
