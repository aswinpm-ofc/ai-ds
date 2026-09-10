# LexiKey — Corpus-Based Predictive Keyboard

A simple Computational Linguistics final project demonstrating:

- Regular expressions
- Corpus cleaning
- Regex tokenization
- Word-frequency analysis
- Unigram language model
- Bigram language model
- Trigram language model
- Prefix matching
- Context-aware predictive suggestions
- Virtual keyboard UI

## 1. Install Python

Use Python 3.10+.

## 2. Create a virtual environment (recommended)

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, you can simply run:

```powershell
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## 3. Install dependencies

```powershell
pip install -r requirements.txt
```

## 4. Run

```powershell
python app.py
```

Open the local address printed by Flask, normally:

http://127.0.0.1:5000

## How it works

The corpus is cleaned using:

```regex
[^a-z\s']
```

Words are extracted using:

```regex
[a-z]+(?:'[a-z]+)?
```

The incomplete word at the end of the user's input is detected using:

```regex
([a-z]+)$
```

The program then combines:

1. prefix matching,
2. trigram context,
3. bigram context,
4. unigram frequency,

to rank the five best suggestions.

## Example

Try:

```text
I want to
```

or:

```text
I am
```

Then type another letter after a space and observe how the suggestion bar changes.

## Viva explanation

The project is a lightweight statistical language model rather than an AI chatbot. It learns word co-occurrence patterns from a fixed corpus. This makes the linguistic process transparent and easy to demonstrate.

## Future improvements

- Larger corpus
- Laplace smoothing
- Keyboard personalization
- Edit-distance spelling correction
- TF-IDF analysis
- POS tagging
- Multilingual corpus
- Evaluation using Top-1 and Top-5 accuracy
