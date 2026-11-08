# ARR Responsible NLP Checklist — 草稿答案

填在 OpenReview 提交表单里，不进 PDF。条目编号按 ARR 2024–2026 版检查表；提交时对照表单实际措辞微调。英文答案可直接粘贴。

## A. For every submission

**A1. Did you describe the limitations of your work?** Yes — Section "Limitations" (scope of items, strength confound and what the bare-assertion control does and does not settle, premise quality and the hand audit, echo protocol, statistics, judging).

**A2. Did you discuss any potential risks of your work?** Yes — Ethics statement: the false notes are released only with their true counterparts and gold labels; the destructive form we identify (a bare shortlist) is already the default in many pipelines, and the contribution is to make it safer.

**A3. Do the abstract and introduction summarize the paper's main claims?** Yes — the five numbered findings in the abstract map one-to-one onto the contributions list and Sections 4–8.

## B. Did you use or create scientific artifacts?

Yes.

**B1. Did you cite the creators of artifacts you used?** Yes — MMLU (Hendrycks et al., 2021) and RACE (Lai et al., 2017) in Section 3; model providers named in Section 3 (DeepSeek V4 Flash/Pro, OpenAI Codex CLI, Kimi K3.1 Pro, Grok 4.7, Qwen2.5-14B-Instruct, Llama-3.1-8B-Instruct, Claude Sonnet/Haiku for two exploratory studies only).

**B2. Did you discuss the license or terms for use of any artifacts?** Partially — MMLU is MIT-licensed; RACE is released for non-commercial research use; both are used for research evaluation only. Model outputs were collected through each provider's public API or CLI under its terms of service. (提交前可在附录加一句 licence 说明。)

**B3. Did you discuss if your use of existing artifacts was consistent with their intended use?** Yes — the benchmarks are used as evaluation items; the released artifact (items, notes, raw responses, pre-registrations) is for research on broadcast robustness.

**B4. Did you discuss the steps taken to check whether the data contains personally identifying information or offensive content?** Items are from public academic benchmarks (MMLU, RACE) and contain no personal data; model-written premises were generated from those items and checked programmatically and by a 40-item hand audit; no offensive content was observed.

**B5. Did you provide documentation of the artifacts?** Yes — Section 3 (items, nodes, protocol, forms) and Appendix A (prompts); the release includes a README mapping each round to its script, pre-registration commit and data directory.

**B6. Did you report relevant statistics like the number of examples, details of train/test/dev splits?** Yes — Section 3 (80-item set: 40 hard + 40 fresh; 297-item set with per-type counts 103/98/96; RACE 51 items); no training, so no splits.

## C. Did you run computational experiments?

Yes.

**C1. Did you report the number of parameters, total computational budget and computing infrastructure?** Partially — API models (DeepSeek, Codex, Kimi, Grok) have undisclosed parameter counts; open-weight nodes are Qwen2.5-14B and Llama-3.1-8B served locally with Ollama on a single Apple-silicon laptop. DeepSeek API spend was small (the control round used 5.3 units of API credit against a pre-registered cap of 15; the 297-item round was of the same order); Codex/Kimi/Grok ran on subscription quotas. (币种按 DeepSeek 后台实际显示填；如果想更完整，附录加一行花费。)

**C2. Did you discuss the experimental setup, including hyperparameter search and best-found hyperparameter values?** Yes — Section 3 "Nodes" gives temperature, system prompt and tool settings per node; there is no hyperparameter search.

**C3. Did you report descriptive statistics about your results (error bars, number of runs)?** Yes — every reported effect carries a 95% bootstrap interval over items (2,000 resamples) and the number of nodes in the same direction; single run per node per item (stated in Limitations "Statistics").

**C4. If you used existing packages, did you report the implementation, model and parameter settings?** Yes — Python with the providers' HTTP APIs / official CLIs; Ollama for the open-weight models; DeepSeek V4 Pro as judge for open-ended answers with the judge prompt in Appendix A.

## D. Did you use human annotators or research with human participants?

**No** crowdworkers or external participants. The only human annotation is the author's own 40-item audit of generated premises (Limitations, "Premise quality"); the audit sheet and marks are in the release. D1–D5 therefore not applicable (answer "No" and state this in the free-text box).

## E. Did you use AI assistants (e.g., ChatGPT, Copilot) in your research, coding or writing?

**Yes — disclose.** Suggested text:

> An AI coding assistant (Anthropic Claude, used through Claude Code) was used throughout: it wrote the experiment scripts and analysis code under the author's direction, drafted the manuscript text, and produced the figure. The author designed the study, set the pre-registered criteria and thresholds, reviewed all code and all numbers, performed the hand audit of premises, and edited and approved every claim in the paper. No Claude model was used as a node in any pre-registered round (Claude Sonnet and Haiku appear only in two exploratory studies, as stated in Section 3). DeepSeek V4 Pro was used to generate the premise notes and to grade open-ended answers, as described in the paper.

## 提交时其它要填的

- 匿名链接：数据代码包（见 ~/Desktop/blind-broadcast-release.zip，上传到 anonymous.4open.science 或作为 supplementary material）。
- 预注册证明：公开仓库里的 commit 哈希（去匿名后给链接）：18fb7799 控制轮、589ba499 297 题轮、481e1bf2 最终形式对照、b5c4e9d7 措辞轮、5cd0304f RACE 外部依据对照、777dcf2b 题型对照。
- Reviewing agreement：第一次投 ARR，按表单提示勾选是否愿意审稿。
