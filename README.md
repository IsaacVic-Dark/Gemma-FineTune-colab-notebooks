# 🤖 Gemma 3 Fine-Tuned Model — AI Integration for Laravel

Fine-tune, load, and export a [Gemma 3 1B](https://huggingface.co/unsloth/gemma-3-1b-it) model using LoRA adapters on Google Colab, then serve it locally via Ollama for seamless HTTP integration with a Laravel backend.

---

## 📁 Notebooks

| Notebook | Purpose |
|---|---|
| `Train InsuOps Model_V2.ipynb` | Fine-tune Gemma 3 on a custom Q&A dataset using LoRA |
| `Load InsuOps Model_V2.ipynb` | Load the fine-tuned model and run inference in Colab |
| `Download InsuOps Model_V2.ipynb` | Merge adapters, convert to GGUF, and download for Ollama |

---

## 🔧 Requirements

- Google Colab (GPU runtime — T4 or better)
- Google Drive (for storing LoRA adapter weights)
- [Ollama](https://ollama.com) installed on your local Windows machine
- Laravel backend with HTTP client (e.g. `Illuminate\Support\Facades\Http`)

---

## 🏋️ 1. Training (`Train InsuOps Model_V2.ipynb`)

Fine-tunes Gemma 3 1B on a custom JSON dataset using Unsloth + LoRA.

**What it does:**
- Uploads and formats a `<your_dataset>.json` file of Q&A pairs
- Loads `unsloth/gemma-3-1b-it` as the base model
- Attaches LoRA adapters (r=16) targeting attention and MLP layers
- Trains for 3 epochs using SFTTrainer with response-only masking
- Saves LoRA adapter weights to Google Drive

**Dataset format** (`<your_dataset>.json`):
```json
[
  {
    "question": "How does <your process> work?",
    "answer": "Here is how <your process> works..."
  }
]
```

**Adapter save location:**
```
Google Drive → MyDrive/<YourModelFolder>/
```

---

## 🔍 2. Loading & Inference (`Load InsuOps Model_V2.ipynb`)

Loads the saved LoRA adapters on top of the base model for testing in Colab.

**What it does:**
- Mounts Google Drive and locates adapter files
- Loads base Gemma 3 model + attaches your LoRA adapters
- Switches to inference mode
- Runs a sample query to verify the model responds correctly

**Test it with:**
```python
question = "Describe how <your workflow> handles <a specific task>."
print(generate_answer(question))
```

---

## 📦 3. Download & Export (`Download InsuOps Model_V2.ipynb`)

Merges LoRA adapters into the base model, converts to GGUF format, and downloads to your Windows machine.

**What it does:**
- Merges adapter weights into the base model via `merge_and_unload()`
- Saves a full merged HuggingFace model (confirms `config.json` exists)
- Converts to GGUF with `Q4_K_M` quantization using llama.cpp
- Downloads the `.gguf` file to your local machine

**Output file location in Colab:**
```
/content/Inc_hf_gguf/Inc_hf.Q4_K_M.gguf
```

---

## 🪟 4. Running in Ollama (Windows)

### Step 1 — Move the GGUF file

Place the downloaded file here:
```
C:\ollama\models\Inc_hf.Q4_K_M.gguf
```
> Create the `C:\ollama\models\` folder if it doesn't exist.

### Step 2 — Create a Modelfile

Create a file called `Modelfile` (no extension) at `C:\ollama\Modelfile`:

```
FROM C:\ollama\models\Inc_hf.Q4_K_M.gguf

PARAMETER temperature 0.7
PARAMETER stop "<end_of_turn>"
PARAMETER stop "<start_of_turn>"

SYSTEM """You are a helpful assistant for <your application>. Answer questions clearly and accurately."""
```

### Step 3 — Register the model

Open PowerShell and run:
```powershell
ollama create <your-model-name> -f C:\ollama\Modelfile
```

### Step 4 — Test the model
```powershell
ollama run <your-model-name> "How does <your process> work?"
```

### Step 5 — Start the API server
```powershell
ollama serve
```
The API will be available at `http://localhost:11434`.

---

## 🔌 5. Laravel Integration

### Simple prompt (generate)
```php
$response = Http::post('http://localhost:11434/api/generate', [
    'model'  => '<your-model-name>',
    'prompt' => 'How does <your process> work?',
    'stream' => false,
]);

$text = $response->json()['response'];
```

### Chat-style (recommended)
```php
$response = Http::post('http://localhost:11434/api/chat', [
    'model'  => '<your-model-name>',
    'stream' => false,
    'messages' => [
        ['role' => 'user', 'content' => 'How does <your process> work?']
    ],
]);

$text = $response->json()['message']['content'];
```

> **Docker users:** Replace `localhost` with `host.docker.internal` — e.g. `http://host.docker.internal:11434`.

---

## 📂 Folder Structure

```
colab-notebooks/
├── Train InsuOps Model_V2.ipynb      # Fine-tuning pipeline
├── Load InsuOps Model_V2.ipynb       # Inference & testing
├── Download InsuOps Model_V2.ipynb   # GGUF export for Ollama
└── README.md
```

---

## 🛠️ Tech Stack

- [Unsloth](https://github.com/unslothai/unsloth) — fast fine-tuning
- [PEFT](https://github.com/huggingface/peft) — LoRA adapter management
- [TRL](https://github.com/huggingface/trl) — SFT training
- [llama.cpp](https://github.com/ggerganov/llama.cpp) — GGUF conversion
- [Ollama](https://ollama.com) — local model serving
- [Laravel HTTP Client](https://laravel.com/docs/http-client) — backend integration
