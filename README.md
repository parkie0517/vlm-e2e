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

## Acknowledgments

Special thanks to the following projects:

- [UniAD](https://github.com/opendrivelab/uniad)
- [Qwen3-VL](https://github.com/QwenLM/Qwen3-VL)
- [DriveLM](https://github.com/OpenDriveLab/DriveLM)
