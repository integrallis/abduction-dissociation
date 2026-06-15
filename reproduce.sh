#!/usr/bin/env bash
# Reproduce the paper's CPU-only findings end to end. No API keys required.
# The live LLM head-to-heads are NOT run here (they need API keys + budget); instead we
# aggregate the CACHED o3 + Sonnet transcripts so the LLM-gap numbers are reproducible too.
# Runtime: ~10-15 min on a laptop CPU.
set -e
cd "$(dirname "$0")"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"

echo "==> tests (the reasoner's invariants)"
python -m pytest tests/ -q

echo "==> S2: ours sits at the recoverability oracle, CA, n=200/rule (Table 'CA scaling')"
python experiments/ca_scaling_ours.py

echo "==> Training-seed sweep: ours == oracle at every (model-init, data) seed, std=0 (Limitations)"
python experiments/ca_seed_sweep.py

echo "==> S3: load-bearing controls -- degrade-to-chance + call-site audit (Table 'controls')"
python experiments/ca_load_bearing.py

echo "==> S4: the inverted wall, seed-robust (Figure 2)"
python experiments/ca_inverted_wall.py

echo "==> Sokoban: abduction size sweep + the enumeration wall (Table 'Sokoban')"
python experiments/sok_size_sweep.py

echo "==> Control axis: ours closes Sokoban control (the matched within-domain comparison, Table 1)"
echo "        ours + random are CPU-deterministic (18/18); cached o3/Sonnet rows aggregate if present"
python experiments/sok_control_llm.py --dump results/sok_control.jsonl --aggregate-only

echo "==> Control axis: the classical A* (true-model) reference on the SAME 18 instances (Table 'control')"
python experiments/sok_control_astar.py

echo "==> S2: the LLM gap + 'widens with horizon' (Figure 1) -- aggregated from the CACHED"
echo "        o3 + Sonnet transcripts in results/ca018.jsonl (no API access needed)"
python experiments/ca_scaling_llm.py --dump results/ca018.jsonl --aggregate-only

echo "==> Graded recovery: re-score cached transcripts by Hamming/rank (collapse, not near-miss)"
python experiments/ca_graded_metric.py --dump results/ca018.jsonl

echo "==> Forward axis: o3 holds forward, the weak model fails (CACHED transcripts)"
python experiments/ca_forward_llm.py --dump results/ca021_forward.jsonl --aggregate-only

echo "==> Paired significance: McNemar exact test, ours vs each LLM on shared instances (no API)"
python experiments/significance.py

echo "==> Tool-augmented: Opus 4.8 + code execution closes the gap (CACHED; the limit is in-context search)"
python experiments/ca_abduction_tool.py --aggregate-only --dump results/ca_opus_tool.jsonl

echo "==> o3 high-effort: the abduction gap is not an effort artifact (CACHED transcripts)"
python experiments/ca_scaling_llm.py --dump results/ca018_high.jsonl --aggregate-only

echo "==> Out-of-range generalization: abduct at L=11, trained only on L<=10"
python experiments/ca_scaling_ours.py --L 11 --n 8 --seeds 1 --noises 0.2

cat <<'EOF'

Done. Everything above ran on CPU with no API access.

To re-run the LIVE LLM head-to-heads yourself (needs ANTHROPIC_API_KEY / OPENAI_API_KEY
in a .env file, and real budget -- o3 is ~3 min/call):
  python experiments/ca_scaling_llm.py --models openai:o3 anthropic:claude-sonnet-4-6 \
      --effort medium --n 15 --horizons 6 12 18 --noise 0.2 --dump results/ca018.jsonl
  python experiments/ca_abduction_llm.py --model openai:o3 --n 4
  python experiments/sok_abduction_llm.py --model openai:o3 --n 8
EOF
