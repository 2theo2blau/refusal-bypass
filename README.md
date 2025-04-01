# Refusal Bypass Tools

This repository contains a collection of tools and techniques for analyzing, testing, and potentially bypassing AI safety guardrails, specifically in the area of refusal behaviors. These tools are intended for **research purposes only** to help understand and improve AI safety mechanisms.

## Overview

The repository includes three main components:

- **Gradient-based Prompt Optimization (GCG)**: An implementation of the "Gradient-based Adversarial Prompt Optimization" technique, as described in the [GCG paper](https://arxiv.org/abs/2307.15043).
- **Refusal Classifier**: A training script for developing models that can detect when an AI system is refusing to answer a query.
- **Prompt Perturbation Methods**: Simple text manipulation techniques that create variations of prompts to test AI system robustness.

## Gradient-based Prompt Optimization (GCG)

The `gcg-prompt.py` file implements the GCG approach from *"Universal and Transferable Adversarial Attacks on Aligned Language Models"* (arxiv:2307.15043). This technique uses gradients to optimize a suffix that can be appended to a user prompt to steer the model toward generating a target output.

### How It Works

1. You provide an input prompt and a target prefix you want the model to output
2. The optimizer creates a random suffix and iteratively improves it to minimize the loss between:
   - What the model would output when given the prompt + suffix
   - Your desired target prefix

The suffix acts like a set of "adversarial instructions" that can potentially override the model's regular behavior.

### Usage

```bash
python gcg-prompt.py --model "model_name" --prompt "your prompt" --target_prefix "desired output" --suffix_len 20 --num_steps 500
```

The script will save suffix optimization progress to a JSON file, tracking the best suffixes found and their associated loss values during the optimization process.

## Refusal Classifier

The `classifier-train.py` script trains a classifier that can detect when a language model is refusing to answer a query versus complying with it.

### Features

- Uses the WildGuardMix dataset with refusal/compliance labels
- Implemented with RoBERTa-large for high-accuracy classification
- Includes comprehensive metrics (accuracy, precision, recall, F1)
- Supports early stopping, mixed precision training, and more

### Usage

```bash
python classifier-train.py
```

You can modify the script's configuration variables to adjust the model type, dataset, or training parameters.

## Prompt Perturbation Methods

`prompt-perturbation.py` implements simpler approaches to generate variations of prompts, which can be useful for testing model robustness against slight changes in input phrasing.

### Techniques:

1. **Synonym Replacement**: Replaces words with semantically similar alternatives, ranked by embedding similarity to preserve context.
2. **Logit-based Substitution**: Uses an auxiliary language model to find token substitutions that have high probability in context.

These techniques can help test model consistency across variations of the same query.

### Usage

```bash
python prompt-perturbation.py
```

## Research Applications

These tools can be used to:

- Test the robustness of AI safety measures
- Discover potential vulnerabilities in current guardrail implementations
- Develop better detection methods for bypass attempts
- Better understand the optimization landscape of large language models

## Ethical Considerations

This repository is provided for research and educational purposes only. The tools should be used responsibly and in accordance with ethical guidelines for AI research. We strongly discourage any malicious applications of these techniques.

## Citation

If using the GCG implementation, please cite the original paper:

```
@misc{zou2023universal,
      title={Universal and Transferable Adversarial Attacks on Aligned Language Models}, 
      author={Andy Zou and Zifan Wang and J. Zico Kolter and Matt Fredrikson},
      year={2023},
      eprint={2307.15043},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
