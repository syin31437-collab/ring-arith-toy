import sys, os
sys.path.insert(0, "llm")
import collect as C
t = open("redteam/theory_packet.txt").read()
import json, urllib.request
e = json.load(open(os.path.expanduser("~/.hermes/auth.json")))["credential_pool"]["deepseek"][0]
body = json.dumps({"model": "deepseek-v4-pro", "max_tokens": 32000, "messages": [{"role": "user", "content": t}]}).encode()
r = urllib.request.Request(e["base_url"].rstrip("/") + "/chat/completions", data=body, headers={"Authorization": "Bearer " + e["access_token"], "Content-Type": "application/json"})
open("redteam/report_deepseek.md", "w").write(json.load(urllib.request.urlopen(r, timeout=1500))["choices"][0]["message"]["content"])
