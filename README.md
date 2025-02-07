# LLM-Supported Natural Language to Bash Translation

## 👋 Overview
This repository contains code for the 2025 NAACL paper:  
[LLM-Supported Natural Language to Bash Translation]()

## 🚀 Quick Start
Note: Our code has only been tested on Ubuntu 20.04 with Python 3.10.
- Install Docker Engine [(Instructions)](https://docs.docker.com/engine/install/)  
- Configure Docker for non-sudo users [(Instructions)](https://docs.docker.com/engine/install/linux-postinstall/) 
- Install Ollama
```bash
curl -fsSL https://ollama.com/install.sh | sh
```
- Install embedding model
```bash
ollama pull mxbai-embed-large
```
- Setup virtual environment
```bash
python3 -m venv nl2sh_venv
source nl2sh_venv/bin/activate
pip install -r requirements.txt
python3 -m ipykernel install --user --name=nl2sh_venv --display-name="nl2sh_venv"
```
- Run example.ipynb

## 🛠️ Repo
- **paper/** - Latex source for our paper
- **example.ipynb** - Starter code
- **model_comparison.ipynb** - Reproduce our best model (+ parser) results
- **finetuned_model_comparison.ipynb** - Reproduce our fine-tuned model results
- **feh_comparison.ipynb** - Reproduce our FEH comparison results

## 🔗 Links
Our datasets, benchmark code and fine-tuned models are available at these links:
- Datasets
  - [NL2SH-ALFA Dataset](https://huggingface.co/datasets/westenfelder/NL2SH-ALFA)
  - [InterCode-Corrections Dataset](https://huggingface.co/datasets/westenfelder/InterCode-Corrections)
- Benchmark
  - [InterCode-ALFA Source Code](https://github.com/westenfelder/InterCode-ALFA)
  - [InterCode-ALFA PyPI Package](https://pypi.org/project/icalfa/)
- Models
  - [Qwen2.5-Coder-0.5B-Instruct-NL2SH](https://huggingface.co/westenfelder/Qwen2.5-Coder-0.5B-Instruct-NL2SH)
  - [Qwen2.5-Coder-1.5B-Instruct-NL2SH](https://huggingface.co/westenfelder/Qwen2.5-Coder-1.5B-Instruct-NL2SH)
  - [Qwen2.5-Coder-3B-Instruct-NL2SH](https://huggingface.co/westenfelder/Qwen2.5-Coder-3B-Instruct-NL2SH)
  - [Qwen2.5-Coder-7B-Instruct-NL2SH](https://huggingface.co/westenfelder/Qwen2.5-Coder-7B-Instruct-NL2SH)
  - [Llama-3.2-1B-Instruct-NL2SH](https://huggingface.co/westenfelder/Llama-3.2-1B-Instruct-NL2SH)
  - [Llama-3.2-3B-Instruct-NL2SH](https://huggingface.co/westenfelder/Llama-3.2-3B-Instruct-NL2SH)
  - [Llama-3.1-8B-Instruct-NL2SH](https://huggingface.co/westenfelder/Llama-3.1-8B-Instruct-NL2SH)
