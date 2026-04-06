# PAVO: Pipeline-Aware Voice Orchestration with Demand-Conditioned Inference Routing

**Authors:** NarasingaMoorthy VeiluKanthaPerumal (University of Pennsylvania), Mohammed Imthathullah (Google)

## Overview

PAVO treats the ASR-LLM-TTS voice pipeline as a jointly optimizable inference graph with demand-conditioned routing. The core contribution is an empirical characterization of **inter-stage coupling constraints** — quality dependencies where upstream ASR configuration bounds feasible downstream LLM options.

## Key Results

- **12% lower P95 latency** vs fixed cloud (p = 2×10⁻⁶) on real H100 GPU experiments
- **34% lower median latency** and **71% lower energy** vs fixed-cloud on 50K-turn benchmark
- **Coupling cliff**: Gemma2 2B accuracy drops from 0.825 to 0.585 at WER=2% (n=200 per level)
- **85K-parameter MLP** meta-controller trained via multi-objective PPO in 106s

## Repository Structure

```
paper/                    # TMLR-formatted LaTeX (preprint + anonymous)
experiments/
  outputs/
    meta_controller.pt           # Trained model weights (85,041 params)
    meta_controller_best.pt      # Best checkpoint
    training_log.json            # PPO training log (100K steps)
    coupling_results_200.json    # Coupling calibration (3,600 LLM calls)
    ablation_bertscore.json      # Real ablation with BERTScore
    ablation_results_real.json   # Ablation with heuristic quality
  exp1_e2e_pipeline.py           # E2E pipeline experiment
  exp2_coupling_calibration.py   # Coupling calibration experiment
  exp3_train_ppo.py              # PPO training experiment
  exp4_fix.py                    # Component ablation (fixed)
tier1_*.json                     # Tier 1 experiment results
tier2_*.json                     # Tier 2 experiment results
tier3_50k_train.jsonl            # PAVO-Bench training set (40K turns)
tier3_50k_test.jsonl             # PAVO-Bench test set (10K turns)
```

## Hardware

- **GPU experiments**: NVIDIA A100-SXM4-40GB (Lambda Labs)
- **M3 experiments**: Apple M3 8GB
- **Models**: Whisper large-v3/tiny, Llama 3.1 8B, Gemma2 2B via ollama

## Citation

```bibtex
@article{veilukanthaperumal2026pavo,
  title={PAVO: Pipeline-Aware Voice Orchestration with Demand-Conditioned Inference Routing},
  author={VeiluKanthaPerumal, NarasingaMoorthy and Imthathullah, Mohammed},
  year={2026}
}
```

## License

CC-BY 4.0

## Links

- **HuggingFace**: [vnmoorthy/pavo-bench](https://huggingface.co/datasets/vnmoorthy/pavo-bench)
