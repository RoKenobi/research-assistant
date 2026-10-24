# Setup

## 1. Install Python dependencies
```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## 2. Check Claude Code
The tool runs every LLM call through Claude Code on your Claude plan; no API key is needed.
```
claude -p "hi"
```
If that fails, run `claude` and `/login`.

## 3. Configure secrets
```
copy .env.example .env
```
Fill in your `.env`:
- `LANGFUSE_PUBLIC_KEY` + `LANGFUSE_SECRET_KEY` + `LANGFUSE_HOST` — from your self-hosted Langfuse (see "Self-hosting Langfuse" in the README), or from a free cloud.langfuse.com account with `LANGFUSE_HOST=https://cloud.langfuse.com`. Optional.
- `GITHUB_TOKEN` — optional, raises GitHub search limits.

## 4. Set your Obsidian vault path in config.yaml
```yaml
vault_path: "C:/Users/YourName/path/to/Obsidian"
```
Check the `lab` section (robots and simulator) while you're there.

## 5. Run
```
python pipeline.py "humanoid whole-body control"
```
Results land in your vault under `Research/humanoid-whole-body-control/`. Drop your own PDFs or a `links.txt` of arXiv links into that topic's `inbox/` folder and the next run reads them.
