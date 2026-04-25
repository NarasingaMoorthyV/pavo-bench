#!/usr/bin/env bash
# Re-apply the repo's About metadata (description, topics, homepage) and
# create/update the v1.0.0 release. Requires a PAT with repo scope in $GH_TOKEN.
set -euo pipefail
: "${GH_TOKEN:?export GH_TOKEN=ghp_... with repo scope first}"

REPO="vnmoorthy/pavo-bench"

curl -fsS -X PATCH \
  -H "Authorization: Bearer $GH_TOKEN" \
  -H "Accept: application/vnd.github+json" \
  "https://api.github.com/repos/$REPO" \
  -d '{
    "description":"A 50K-turn voice pipeline benchmark and an 85K-param meta-controller that cuts P95 latency 10.3% and energy 71% vs fixed cloud. TMLR 2026.",
    "homepage":"https://huggingface.co/datasets/vnmoorthy/pavo-bench",
    "has_issues":true,
    "has_wiki":false
  }' | python3 -c 'import json,sys;d=json.load(sys.stdin);print("description set:", d.get("description","")[:80])'

curl -fsS -X PUT \
  -H "Authorization: Bearer $GH_TOKEN" \
  -H "Accept: application/vnd.github+json" \
  "https://api.github.com/repos/$REPO/topics" \
  -d '{"names":["voice-ai","asr","llm","tts","benchmark","inference-optimization","edge-ai","latency","reinforcement-learning","ppo","speech-recognition","whisper","llama","gemma","voice-assistant","machine-learning","deep-learning","pytorch","tmlr"]}' \
  | python3 -c 'import json,sys;d=json.load(sys.stdin);print("topics set:", ", ".join(d.get("names",[])))'

echo "Metadata applied."
