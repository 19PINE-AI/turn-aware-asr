# Citation Verification — Appendix A

Verified 2026-05-19. Sources: arxiv.org, github.com/kyutai-labs, kyutai.org, huggingface.co, daily.co, blog.livekit.io.

## 1. Kyutai STT (Kyutai Labs, 2025)

| Field | Value |
|---|---|
| Claimed ID | n/a (project, not paper) |
| Verdict | Real |
| Real ID | github.com/kyutai-labs/delayed-streams-modeling; kyutai.org/stt |
| Summary | Kyutai STT is a family of streaming speech-to-text models built on the Delayed Streams Modeling (DSM) framework. Released models include `stt-1b-en_fr` (~1B params, 0.5s delay, with a semantic VAD head) and `stt-2.6b-en` (~2.6B params, 2.5s delay). The semantic VAD predicts end-of-turn from streamed audio and is currently exposed via the Rust server. |
| Relevance | Closest open analog to Bo's plan: same DSM idea on Mimi-tokenized streams plus a semantic VAD head. Direct architectural reference for the joint VAD+ASR objective. |

## 2. Mini-Omni / Mini-Omni2 (Tsinghua, 2024)

| Field | Value |
|---|---|
| Claimed ID | n/a |
| Verdict | Real |
| Real ID | arXiv:2408.16725 (Mini-Omni); arXiv:2410.11190 (Mini-Omni2) |
| Summary | Mini-Omni is an end-to-end streaming speech-in/speech-out conversational LLM built on a Qwen2-0.5B backbone with a Whisper-small encoder and a parallel text+audio token generation paradigm. Mini-Omni2 extends this to vision and adds duplex interaction with command-based interruption. Both demonstrate that a 0.5B base plus modest synthetic data can drive real-time speech interaction. |
| Relevance | Recipe-level reference for fitting a small (Qwen3-0.6B) backbone with streaming audio tokens; informs adapter design and parallel text-audio decoding for Bo's Mimi+Qwen3 stack. |

## 3. Moshi (Kyutai, 2024)

| Field | Value |
|---|---|
| Claimed ID | n/a |
| Verdict | Real |
| Real ID | arXiv:2410.00037 |
| Summary | Moshi is a full-duplex speech-text foundation model combining the Helium text LLM, the Mimi streaming neural audio codec (12.5 Hz, 1.1 kbps, 80 ms latency), and a multi-stream hierarchical token generator. It models the user and agent audio streams jointly plus an "inner monologue" text stream, achieving ~200 ms real-time latency. |
| Relevance | Source of the Mimi codec and the delayed-streams text/audio interleaving that Bo's plan adopts wholesale; defines the token rate and frame structure for the VAD+ASR streams. |

## 4. Uni-ASR — arXiv:2603.11123

| Field | Value |
|---|---|
| Claimed ID | 2603.11123 |
| Verdict | Real (confirmed despite suspicion) |
| Real ID | arXiv:2603.11123 (submitted 2026-03-11; targeted at Interspeech 2026) |
| Summary | "Uni-ASR: Unified LLM-Based Architecture for Non-Streaming and Streaming ASR" (Xia, Tang, Hou, Xu, Yao). Proposes a single LLM-based ASR that joint-trains streaming and non-streaming modes with a context-aware fallback decoding strategy to recover streaming accuracy without added latency. |
| Relevance | Directly informs Bo's "one model, two modes" training plan; the joint streaming/non-streaming objective and fallback decoder are reusable for the streaming VAD+ASR head. |

## 5. BR-ASR — arXiv:2505.19179

| Field | Value |
|---|---|
| Claimed ID | 2505.19179 |
| Verdict | Real |
| Real ID | arXiv:2505.19179 (Interspeech 2025) |
| Summary | "BR-ASR: Efficient and Scalable Bias Retrieval Framework for Contextual Biasing ASR in Speech LLM" (Gong, Lv, Wang, Zhu, Qian). Scales contextual biasing to ~200k bias entries via contrastive speech-bias retrieval plus curriculum learning to reduce homophone confusion, with minimal added latency. |
| Relevance | Provides a scalable retrieval pre-stage that fits Bo's "text prefix biasing" injection point; useful when the hotword set is too large to fit in prompt context. |

## 6. "Contextual Biasing for LLM-Based ASR with Hotword Retrieval and RL" — arXiv:2512.21828

| Field | Value |
|---|---|
| Claimed ID | 2512.21828 |
| Verdict | Real |
| Real ID | arXiv:2512.21828 (submitted 2025-12-26) |
| Summary | Kong et al. propose a two-stage pipeline: a GLCLAP-based retriever returns top-k hotword candidates from large vocabularies, then an LLM-ASR is fine-tuned with GRPO using a task-driven reward that injects the retrieved hotwords as a textual prompt. Reduces keyword error rate without hurting general ASR. |
| Relevance | Validates Bo's exact mechanism (text prefix biasing + retrieved hotwords) and adds an RL objective (GRPO) that could be layered on top of supervised fine-tuning. |

## 7. FastTurn-Unified — arXiv:2604.01897

| Field | Value |
|---|---|
| Claimed ID | 2604.01897 (cited as "FastTurn-Unified") |
| Verdict | Mis-cited title (ID and work are real) |
| Real ID | arXiv:2604.01897, actual title "FastTurn: Unifying Acoustic and Streaming Semantic Cues for Low-Latency and Robust Turn Detection" (Wang, Xue, He, et al.; v4 2026-04-27) |
| Summary | FastTurn combines streaming CTC decoding over acoustic features with semantic cues to enable early end-of-turn decisions from partial observations, targeting full-duplex agents. Releases a human-dialogue test set and outperforms baselines under noisy conditions. |
| Relevance | Direct competitor/benchmark for Bo's semantic-VAD head; the acoustic+semantic fusion design is a strong baseline to compare the Mimi-token-based VAD against. |

## 8. LiveKit Smart Turn v2 (2025)

| Field | Value |
|---|---|
| Claimed ID | n/a |
| Verdict | Mis-attributed |
| Real ID | "Smart Turn v2" is by Pipecat / Daily (huggingface.co/pipecat-ai/smart-turn-v2); LiveKit ships a separate `livekit/turn-detector` (Qwen2.5-0.5B-Instruct distilled), most recently v0.4.1-intl. |
| Summary | Pipecat Smart Turn v2 is an open-source semantic-VAD model that operates on raw waveform via wav2vec2 + linear head, outputs an end-of-turn probability, supports 14 languages, and reports ~99% accuracy on `human_5_all`. LiveKit's own turn detector is a text-only transformer fine-tuned from Qwen2.5-0.5B-Instruct and is distinct from Smart Turn v2. |
| Relevance | Both are open semantic-endpointing baselines for Bo's VAD head. Citation must be split: either cite Pipecat Smart Turn v2 (audio-based) or LiveKit turn-detector (text-based), not "LiveKit Smart Turn v2". |

## Citation IDs/labels that must be fixed in the plan

- "LiveKit Smart Turn v2 (2025)" — mis-attributed; rename to "Pipecat Smart Turn v2 (Daily, 2025)" and/or add a separate entry for "LiveKit turn-detector (Qwen2.5-0.5B, 2025)".
- "FastTurn-Unified (arXiv 2604.01897)" — correct ID, but the actual title is "FastTurn: Unifying Acoustic and Streaming Semantic Cues for Low-Latency and Robust Turn Detection"; update the label.
- Optional: add explicit arXiv IDs for Mini-Omni (2408.16725), Mini-Omni2 (2410.11190), and Moshi (2410.00037) to remove ambiguity.

All four suspicious arXiv IDs (2603.11123, 2505.19179, 2512.21828, 2604.01897) resolved to real papers on arxiv.org.
