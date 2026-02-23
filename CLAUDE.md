# Speech-Act Pilot Experiment

## Research Question

Does changing ONLY the speech-act style of tool feedback (diagnostic / directive /
suggestive / accusatory / neutral) — with IDENTICAL factual content — systematically
change LLM agent recovery behavior?

## Role Boundary (CRITICAL)

- CC1 = 实现 + 修 bug。**不要自己做 code review**。
- Code review 是 CC2 的职责。CC1 写完代码后提 PR，等用户让 CC2 review。
- 不要使用 code-reviewer agent 自查代码。Smoke test 可以做，review 不行。

## Workflow

1. Receive plan → break into modules (1 module = 1 PR)
2. Implement on feature branch (`speech_act/<module>-<description>`)
3. PR → **等用户安排 CC2 review** → 根据 CC2 反馈修 → merge → next module
4. Use `/plan` before starting each PR, `/tdd` before writing code

## Commit Format

`<type>: <description>` — types: feat, fix, refactor, exp, data, docs, test

## Go/Kill Criteria

| Phase | Kill If | Go If |
|-------|---------|-------|
| B1 (minimal cue) | Chi-square p > 0.1 across 5 styles | Any style pair significantly differs |
| B2 (natural lang) | Same as B1 | Effect size > B1 |
| M1 (mechanism) | Attention probes noisy / uninterpretable | Clear style→action mapping in specific heads |

## Environment

- **Model**: Qwen2.5-7B-Instruct (local: `/E2M181-data/zhangcb/models/Qwen2.5-7B-Instruct/`)
- **Conda env**: `agent-robustness` (Python 3.10, has transformers, torch, tiktoken, scipy, vllm, accelerate)
- **Python**: `/home/zhangcb/miniconda3/envs/agent-robustness/bin/python`
- **GPU**: 3× RTX A5000 (23GB each). Use GPU 2 (index 2) — least loaded. Set `CUDA_VISIBLE_DEVICES=2`.
- **Serving**: vLLM on localhost:8001

## Serving the Model (vLLM)

```bash
CUDA_VISIBLE_DEVICES=2 python -m vllm.entrypoints.openai.api_server \
    --model /E2M181-data/zhangcb/models/Qwen2.5-7B-Instruct \
    --dtype auto \
    --max-model-len 4096 \
    --port 8001 \
    --gpu-memory-utilization 0.85
```

## Project Layout

```
├── mock_tools.py          # Deterministic mock tool environment
├── renderer.py            # Speech-act style renderer (B1 minimal + B2 natural)
├── tasks.py               # 5 task scenarios with error→recovery paths
├── agent.py               # ReAct agent loop (vLLM-served Qwen2.5-7B)
├── run_experiment.py       # Experiment runner (5×5×20 = 500 runs)
├── analyze.py             # Statistical analysis + plots
├── requirements.txt       # Pinned dependencies
├── tests/                 # Tests for each module
└── results/               # Output (gitignored)
```

## PR Sequence

1. `speech_act/mock-tools-renderer` — mock_tools.py + renderer.py + token count tests
2. `speech_act/agent-tasks` — tasks.py + agent.py + smoke test
3. `speech_act/experiment-runner` — run_experiment.py + analyze.py
4. `speech_act/b1-results` — B1 raw results + analysis output

## Implementation Principles

### Token Count Matching (NON-NEGOTIABLE)
- Use tiktoken to verify ALL style renderings for the same task have EXACTLY the same token count
- If counts don't match, the experiment is invalid
- Write automated tests that assert token parity

### No Information Leakage
- Style templates must NOT introduce facts beyond what's in error state S
- "You should have known" (accusatory) = pragmatic frame, not new information

### Action Classification Must Be Rule-Based
- Classify agent actions via string matching on action + params
- Categories: retry, modify_params, switch_tool, ask_user, give_up, correct_recovery
- NO LLM-based classification

### Phased Execution
- B1 first. Do NOT start B2 until B1 results are analyzed and reviewed
- Each phase = separate PR for review

## Research Code Rules

- **Reproducibility**: pin deps, set seeds, log configs
- **Kill test mindset**: speed over elegance, clear go/kill criteria, always include baseline
- **Self-check before PR**: smoke test passes, no hardcoded paths, outputs gitignored

## Pre-PR Checklist

- [ ] Smoke test passes
- [ ] No hardcoded absolute paths (model path is the exception)
- [ ] Results directory gitignored
- [ ] Seeds set and documented
- [ ] Token counts verified (renderer)
