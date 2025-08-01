# Multilingual Logistics Summarization Benchmark

This repository contains the evaluation benchmark and dataset presented in our paper: **"SYNAPSE: Synthesizing Narratives from Agentic Path Spatial Exploration"**.

This benchmark is designed to facilitate research in the automated generation of multilingual, human-centered summaries for logistics operations. It provides a high-quality, human-validated dataset for evaluating systems that can synthesize multimodal data (e.g., geospatial coordinates, failure predictions, workload data) into coherent, multilingual narratives.

## Dataset Description

The benchmark consists of 100 semi-synthetic logistics routes and the corresponding natural language queries and golden summaries.

*   **`data/emails.json`**: Contains 100 natural language queries used as inputs for an evaluation. Each JSON object includes a user query (e.g., "Briefing for Route-001 tomorrow") and the `expected_route` name that a Natural Language Understanding (NLU) system should extract.

*   **`data/golden_summaries.csv`**: This is the core of the benchmark. It contains the human-authored and validated "golden" standard summaries for each of the 100 routes. For each route, the file provides:
    *   A canonical **English summary**, synthesized from underlying route data and formatted according to a consistent standard.
    *   Expert, native-speaker-validated translations of the English summary into **Spanish**, **French**, **Italian**, **Portuguese**, and the low-resource language **Luxembourgish**.

*   **`data/routes.csv`**: Contains the raw latitude/longitude coordinates and stop sequence for each of the 100 routes.

## How to Use This Benchmark

This dataset is designed to be used as a ground truth for evaluating generative AI systems that perform route summarization and translation. A typical workflow would be:

1.  **Input:** Use the queries from `data/emails.json` as the input prompts for your system.
2.  **Output:** Generate a multilingual summary for each route using your own model or framework.
3.  **Evaluate:** Compare your system's generated output against the corresponding golden summaries in `data/golden_summaries.csv`.

Standard translation quality metrics such as **COMET**, **METEOR**, and **TER** can be used to quantitatively measure performance against this benchmark. The inclusion of English allows for the evaluation of a system's ability to adhere to a specific structured narrative format, while the other languages test for translation quality.
