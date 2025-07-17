# Fine-tuning Quick Start

## Server Setup
```bash
# DigitalOcean NVIDIA H100 recommended
ssh -A root@
ssh-add -l
# check python version
apt install tmux nvtop git vim git-lfs python3.10-venv jq
curl -sS https://starship.rs/install.sh | sh
echo 'eval "$(starship init bash)"' >> ~/.bashrc
curl -fsSL https://ollama.com/install.sh | sh
```

## Python Venv Setup
```bash
python3 -m venv ft_venv
source ft_venv/bin/activate
pip install -r requirements.txt
# print(torch.cuda.is_available())
```

## HuggingFace + WandB Setup
```bash
huggingface-cli login
wandb login
huggingface-cli lfs-enable-largefiles .
huggingface-cli download Qwen/Qwen2.5-Coder-3B-Instruct
huggingface-cli scan-cache
huggingface-cli repo create NL2SH --type model # make sure this matches .py
```

## Train
```bash
tmux new -s ft
source ft_venv/bin/activate
# check run number and paths in .py
python finetune.py > finetune.log 2>&1 & disown
tail -f finetune.log
nvtop
```

## Convert to GGUF
```bash
git clone https://github.com/ggerganov/llama.cpp.git
python3 -m venv gguf_venv
source gguf_venv/bin/activate
pip install -r llama.cpp/requirements.txt
python llama.cpp/convert_hf_to_gguf.py -h
python llama.cpp/convert_hf_to_gguf.py final/ --outfile NL2SH.gguf --outtype bf16
```

## Upload to Ollama
```bash
sudo cat /usr/share/ollama/.ollama/id_ed25519.pub
# Add public key to website
# Verify Modelfile is correct
ollama create westenfelder/NL2SH
ollama push westenfelder/NL2SH
```

## Cleanup
```bash
git status
rm -rf wandb/ checkpoints/ final/ finetune.log NL2SH.gguf llama.cpp/
```
