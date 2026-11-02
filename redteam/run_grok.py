import json, os, urllib.request
e = json.load(open(os.path.expanduser("~/.hermes/auth.json")))["credential_pool"]["xai"][0]
print({k: (v if k in ("base_url", "auth_type", "source") else "…") for k, v in e.items() if k in ("base_url", "auth_type", "source", "access_token")})
t = open("redteam/theory_packet.txt").read()
body = {"model": "grok-4.7", "max_tokens": 16000, "messages": [{"role": "user", "content": t}]}
r = urllib.request.Request((e.get("base_url") or "https://api.x.ai/v1").rstrip("/") + "/chat/completions", data=json.dumps(body).encode(), headers={"Authorization": "Bearer " + e["access_token"], "Content-Type": "application/json", "User-Agent": "ring-arith-toy"})
d = json.load(urllib.request.urlopen(r, timeout=1500))
open("redteam/report_grok.md", "w").write(d["choices"][0]["message"]["content"])
print("ok", d.get("usage"))
