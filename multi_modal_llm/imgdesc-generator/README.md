# imgdesc-generator
The `make_data.py` script takes images with defects and prompts Google Gemini and Anthropic Claude Haiku to generate a description of the defect.

## Setup
Add the appropiate API keys into `.env.example` and rename it to `.env`.
Additionally make sure that you have the dependencies installed using
```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
