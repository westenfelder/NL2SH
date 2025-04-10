# LLM-Supported Natural Language to Bash Translation

## 👋 Overview
This repository contains code for the 2025 NAACL paper:  
[LLM-Supported Natural Language to Bash Translation](https://arxiv.org/abs/2502.06858)  

**TLDR:** Large language models (LLMs) are unreliable at translating natural language to Bash commands (NL2SH). We present methods to measure and improve the NL2SH performance of LLMs.

## 🚀 Quick Start
1. Install Ollama
```bash
curl -fsSL https://ollama.com/install.sh | sh
```
2. Download our model
```bash
ollama pull westenfelder/NL2SH
```
3. Add a shortcut function to .bashrc
```bash
nlsh() {
  local prompt="$1"
  curl -s localhost:11434/api/generate -d "{\"model\": \"westenfelder/NL2SH\", \"prompt\": \"$prompt\", \"stream\": false}" | jq -r '.response'
}
```
4. Query the model
```bash
nlsh "print num python files in the top level of the current dir"
```
5. Run the commands at your own risk!

## ⚙️ Full Setup
Note: Our code has only been tested on Ubuntu 20.04 with Python 3.10 and PyTorch 2.6.0+cu124.

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
- Start by running example.ipynb

## 🛠️ Repo Structure
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

## ✍️ Citation
If you find our work helpful, please cite:
```
@misc{westenfelder2025llmsupportednaturallanguagebash,
      title={LLM-Supported Natural Language to Bash Translation}, 
      author={Finnian Westenfelder and Erik Hemberg and Miguel Tulla and Stephen Moskal and Una-May O'Reilly and Silviu Chiricescu},
      year={2025},
      eprint={2502.06858},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2502.06858}, 
}
```
