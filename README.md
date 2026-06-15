# Abduction is the axis

Reproduction code for **“Abduction: a reasoning axis where learned simulation and unaided frontier LLMs dissociate”** ([`paper/main.pdf`](paper/main.pdf)).

**The finding.** *Abduction* — recovering an unobserved initial condition from a noisy trajectory — is a
distinct axis of reasoning, alongside forward execution and goal-directed control. A deliberately
non-transformer reasoner that learns a world model from black-box `(state, [action,] next-state)`
transitions (it never reads the true rule) and reasons by *simulating* that model exposes a
**forward/control-pass, inverse-fail dissociation** in *unaided* frontier LLMs: a dedicated reasoning model
(o3) *closes* forward execution and interactive control, but **does not close abduction** — and the gap
*widens* with difficulty, **cross-domain** (a reversible 1-D cellular automaton and an irreversible 2-D
Sokoban), with the learned model proven load-bearing. The failure is one of *unaided* inference: given
code execution the same model writes the search and closes the gap, which localizes the limitation to
in-context exact inverse search rather than reasoning in general. The inference itself is classical
**simulation-based inference** (MAP over forward rollouts); the contribution is the *black-box-learned*
substrate and the *capability dissociation*, not the inference method.

## Reproduce in one command (CPU-only, no API keys, ~15–20 min)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
./reproduce.sh
```

`reproduce.sh` runs the tests and every CPU-only result, and aggregates the **cached** LLM
transcripts (o3, Sonnet, GPT-4o-mini, and Opus 4.8, in `results/*.jsonl`) so every LLM-gap number
reproduces **without any API access**. The same CPU-only path runs in CI on every push
(`.github/workflows/ci.yml`), and a [`Dockerfile`](Dockerfile) pins a clean environment
(`docker build -t abduction-dissociation . && docker run --rm abduction-dissociation`).

## What each paper result maps to

| What it shows (where in the paper) | Command | Expected (key numbers) |
|---|---|---|
| Ours sits at the recoverability **oracle** — CA, n=200/rule, multi-seed + CIs (Table 4) | `python experiments/ca_scaling_ours.py` | `ours == oracle` at every (rule, noise); e.g. Rule 110 noise 0.2 → **0.965**, Rule 90 → 0.865 |
| The LLM gap **widens with horizon** — CA, o3 + Sonnet (**Figure 5**; per-rule in Table 2) | `python experiments/ca_scaling_llm.py --dump results/ca018.jsonl --aggregate-only` | H=6/12/18: ours 0.47/0.73/**1.00**, o3 0.10/0.17/**0.03**, Sonnet ~0; pooled H=18 ours 30/30 vs o3 1/30 vs Sonnet 0/30 |
| **Graded recovery** — collapse, not near-miss; re-scores the cached transcripts, no new calls (CA Results §3.1) | `python experiments/ca_graded_metric.py --dump results/ca018.jsonl` | LLMs ≈3–4 bits wrong at every H (chance 8), flat as ours → 0 Hamming; oracle posterior on true IC rises 0.34→0.75→**0.99** while LLMs stay ~4 bits off; self-validates on all 180 rows |
| Load-bearing controls — **degrade-to-chance** + **call-site audit** (**Figure 6**, Table 4) | `python experiments/ca_load_bearing.py` | recovery tracks fidelity 0.00→0.017→0.033→**0.95**; learned-rule ticks **11**, true-CA ticks **0** |
| The **inverted wall** — irreducibility aids the inverse (**Figure 7**) | `python experiments/ca_inverted_wall.py` | Rule 110 likelihood gap > Rule 90 at every H (worst seed); 0.019→**0.095** vs 0.008→0.036 |
| Sokoban abduction **size sweep** + the **enumeration wall** (Table 3) | `python experiments/sok_size_sweep.py` | `ours == oracle` 30/36, 33/36, 30/36 (6/7/8); wall: 8×8 = 14,880 (fits) → 10×10 = 97,527 → 16×16 = 2.8M (past cap) |
| **Forward axis** — o3 holds forward, the weak model fails (Table 1; cached) | `python experiments/ca_forward_llm.py --dump results/ca021_forward.jsonl --aggregate-only` | ours **1.00** at every N; o3 exact to N=16, **3/5 at N=32** (Rule 110); GPT-4o-mini **0/5 from N=4** |
| **Second reasoning model** — Opus 4.8 closes forward, does not reliably abduct (Results §3.1, Limitations; cached) | `python experiments/ca_forward_llm.py --dump results/ca_opus_forward.jsonl --aggregate-only` (forward); `python experiments/ca_abduction_anthropic.py --aggregate-only --dump results/ca_opus_abduction.jsonl` (abduction) | forward **12/12** (N=16, N=32); abduction pooled **10/30** vs ours 21/30 (n=5/cell), gap widens 0→+4 (R110) over horizon; **0 truncation drops** |
| **Tool-augmented closes the gap** — the limitation is in-context exact search, not reasoning (Results "Is the gap just in-context search?") | `python experiments/ca_abduction_tool.py --aggregate-only --dump results/ca_opus_tool.jsonl` | Opus 4.8 **+ code execution** recovers **20/30** (≈ ours 21/30), ran code 30/30 — vs **10/30** tool-free; gap closes |
| **Control axis** — o3 *closes* control where it fails abduction, the matched comparison (Table 1) | `python experiments/sok_control_llm.py --cpu-only` (ours/random, CPU-free); `python experiments/sok_control_astar.py` (A* reference) | ours **18/18**, A* (true model) **18/18**, random 10/18 (6/6 → 0/6 by difficulty); o3 **17/18**, Sonnet 10/18 from a live run (see below) |
| **o3 high-effort** — the gap is not an effort artifact (CA Results, in text; cached) | `python experiments/ca_scaling_llm.py --dump results/ca018_high.jsonl --aggregate-only` | o3@high **4/30** (vs 1/30 medium); parse **27/30**; **0/15** on irreducible Rule 110 |
| **Out-of-range generalization** — abduct at unseen widths (CA Results, in text) | `python experiments/ca_scaling_ours.py --L 11` (and `--L 12`) | trained on L≤10; `ours == oracle` at L=11 (**57/60, 55/60**) and L=12 (58/60, 49/60) |
| **Training-seed robustness** — `ours == oracle` is not a seed-0 artifact (Limitations) | `python experiments/ca_seed_sweep.py` | ours = oracle at **every** of 5 (model-init, data) seeds: 56/60 (R110), 47/60 (R90), **std = 0.000** |
| **Paired significance** — McNemar exact test on shared instances (CA Results) | `python experiments/significance.py` | ours vs o3 **p=1.4e-17**, vs Sonnet **5.4e-20**, vs Opus 4.8 **9.8e-4**; ours never loses a discordant instance (c=0) |

Numbers are stochastic only in the LLM rows; the CPU rows are deterministic given the seeds in each
script. Small per-run variation in the multi-seed tables is within the reported CIs.

## Live LLM head-to-heads (optional — need API keys + budget)

The cached transcripts make the paper's LLM numbers reproducible without spending anything. To re-run the
live head-to-heads, put `ANTHROPIC_API_KEY` / `OPENAI_API_KEY` in a `.env` file (gitignored). **o3 is slow
(~3 min/call) and costs real money.**

```bash
python experiments/ca_scaling_llm.py --models openai:o3 anthropic:claude-sonnet-4-6 \
    --effort medium --n 15 --horizons 6 12 18 --noise 0.2 --dump results/ca018.jsonl   # resumable
python experiments/ca_abduction_llm.py  --model openai:o3 --n 4   # CA abduction head-to-head (+ --dump)
python experiments/sok_abduction_llm.py --model openai:o3 --n 8   # Sokoban abduction head-to-head
python experiments/sok_control_llm.py --models openai:o3 anthropic:claude-sonnet-4-6 \
    --n 6 --dump results/sok_control.jsonl                        # Sokoban CONTROL (the matched axis); resumable
python experiments/ca_abduction_anthropic.py --model claude-opus-4-8 --effort medium \
    --n 5 --horizons 6 12 18 --noise 0.2 --dump results/ca_opus_abduction.jsonl   # 2nd reasoning model
python experiments/ca_abduction_tool.py --n 5 --horizons 6 12 18 --noise 0.2 \
    --dump results/ca_opus_tool.jsonl     # tool-augmented (Opus 4.8 + code execution); ~$0.05/call, gap closes
#   ^ truncation-aware: Opus's adaptive-thinking trace counts against the token cap, so this runner
#     uses the 128k ceiling, logs stop_reason, scores only completed calls, and self-heals (retries at
#     lower effort) on the rare instance whose thinking overruns 128k -- no instance is silently dropped.
```

## Repository layout

```
pce/                      the reasoner (package name = "Parallel Counterfactual Evaluation")
  ca/   env, rule (learned local rule), estimate (MAP abduction), planner, headtohead (LLM baseline)
  sok/  env, rule, abduce (MAP over candidate boards), planner (control), headtohead
  llm.py                  live Anthropic/OpenAI/Ollama client for the LLM baselines
experiments/              one script per paper result (named, not numbered)
tests/                    the reasoner's invariants (run by reproduce.sh)
results/*.jsonl           cached o3 / Sonnet / GPT-4o-mini / Opus-4.8 transcripts (every LLM number reproduces)
paper/                    NeurIPS 2026 LaTeX source + the compiled main.pdf
  data/*.csv              figure data (fig_widens/wall/degrade read these; provenance noted in each .tex)
```

## The reasoner, in two sentences

One weight-shared **radius-1 local rule** (a small MLP over a cell's neighbourhood, plus the action when the dynamics are controlled) is
trained on black-box transitions and applied at every cell and every system size, so it size-generalizes
**by construction**. Abduction is a **MAP estimate**: enumerate candidate initial conditions, roll each
forward under the *learned* rule, and keep the one whose trajectory best matches the noisy observation —
classical simulation-based inference, with the forward model learned rather than given.

## Scope and caveats

This is a small-scale study on abstract, vision-free toy domains (L=8 CA; 6×6–8×8 Sokoban), scored by exact
bit/state equality — **not SOTA, not an LLM replacement.** o3 is *not* weak in general: it closes forward
execution and control (Sokoban control 17/18) and partially abducts; the precise claim is that it does not
*reliably* abduct and decays as the trajectory grows. MAP enumeration is feasible only at small state
spaces — `sok_size_sweep.py` and the CA wall (`ca_scaling_ours.py` past L≈12) **measure** where it
ends, and amortized inference is the next step.

## Citation

```bibtex
@misc{abduction_dissociation_2026,
  title  = {Abduction: a reasoning axis where learned simulation and unaided frontier LLMs dissociate},
  author = {Sam-Bodden, Brian},
  year   = {2026}
}
```

## License

MIT — see [`LICENSE`](LICENSE).
