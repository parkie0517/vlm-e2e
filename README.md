# VLM-Based End-to-End Autonomous Driving: Comparative Analysis of Reasoning Strategies and Action Heads

> This project is an ongoing research project.

## Project Overview

This project compares and analyzes end-to-end autonomous driving performance by combining various reasoning strategies and action heads for vision-language models (VLMs), based on the Intern v2 model and the DriveLM-nuScenes dataset.

## Overview Image

![Project Overview](./src/overview.png)

## My Contributions

- Fine-tuning using only Vision Language Adapters and LoRA instead of full fine-tuning
- Prompt engineering with three Chain-of-Thought strategies
- Action head design and comparison: MLP, Diffusion, and RT-2-style tokenization
- Data loader design for VLM-based end-to-end autonomous driving
- Quantitative and qualitative evaluation, including L2 error, collision rate, and failure case analysis

## Results

- Preparation of a paper-style report for beginners in VLM-based end-to-end autonomous driving
- Quantitative comparative analysis of the contributions of VLM reasoning strategies and action heads
- Release of an open-source codebase for future research

## To Do

### Build Dataset

- [ ] Create a concrete plan for this project: `heejun/plan.txt`
- [ ] Align DriveLM v1.1 and UniAD trajectory labels
- [ ] Build a dataset class (loader) that unifies DriveLM and UniAD-style data
- [ ] Verify one aligned sample by visualizing the ground-truth trajectory on the front camera image (front image, Q&A entry, GT)

### Build Baseline Model (No CoT + MLP Head)

- [ ] Build the baseline model: VLM model with LoRA on the LLM (optional: vision adapter)
- [ ] Attach an MLP head (LLM features -> MLP head -> trajectory output)
- [ ] Run a forward pass and verify output shape, loss computation, and metric pipeline with visualization
- [ ] Train and evaluate on a small subset first (L2 loss only; evaluate with L2, ADE, FDE, `obj_col`, and `obj_box_col`)
- [ ] Train and evaluate on the full dataset

### Create and Train CoT Pipeline Using Baseline Method

- [ ] Implement a simple CoT pipeline
- [ ] Train and evaluate on a small subset (L2 loss only)
- [ ] Train and evaluate on the full dataset
- [ ] Implement a DriveLM-style CoT pipeline
- [ ] Train and evaluate on a small subset, first with trajectory-only loss (optional: DriveLM-style language loss)
- [ ] Train and evaluate on the full dataset

### Analyze MLP Head Method

- [ ] Compare three variants: no CoT, simple CoT, and DriveLM-style CoT

### Implement RT-2 Head

- [ ] Implement the RT-2 tokenization head
- [ ] Train and test with three CoT variants on a subset
- [ ] Train and test with three CoT variants on the full dataset

### Analyze Token Head Method

- [ ] Compare three variants: no CoT, simple CoT, and DriveLM-style CoT

### Implement Diffusion Head

- [ ] Implement the diffusion head
- [ ] Train and test with three CoT variants on a subset
- [ ] Train and test with three CoT variants on the full dataset

### Analyze Diffusion Head Method

- [ ] Compare three variants: no CoT, simple CoT, and DriveLM-style CoT

## Acknowledgments

Special thanks to the following projects:

- [UniAD](https://github.com/opendrivelab/uniad)
- [Qwen3-VL](https://github.com/QwenLM/Qwen3-VL)
- [DriveLM](https://github.com/OpenDriveLab/DriveLM)
