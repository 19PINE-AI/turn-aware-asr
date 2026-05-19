# Recent SOTA Scan (Nov 2025 - May 2026)

Scope: streaming ASR with LLM backbones, semantic endpointing, context biasing, ASR-friendly codecs, and 2026 production systems, vs Bo's plan (Kyutai delayed streams + Qwen3-0.6B + Mimi + text-prefix biasing). Dates from arXiv or vendor posts.

---

## 1. Streaming ASR with LLM backbones

- **Qwen3-ASR Technical Report** (arXiv:2601.21337, Feb 2026). 0.6B and 1.7B; 128-dim FBank at 12.5 Hz with 8x downsampling; FlashAttention with **dynamic 1-8 s window** for unified streaming+offline in a single checkpoint. 52 languages. Includes a RAG hotword mechanism (phoneme/word-piece lexicon). **Beat/learn:** unified streaming+offline at 12.5 Hz is directly comparable to Bo's Mimi-rate plan; dynamic-window attention is cleaner than a fixed 1 s text delay.

- **Streaming SR with Decoder-Only LLMs and Latency Optimization** (arXiv:2601.22779, Jan 2026, Wan et al.). Read/write policy net with monotonic chunkwise attention (MoChA), minimal-latency loss, weight-shared non-streaming twin. 5.1%/5.5% CER on Mandarin, 62.5% token-delay reduction. **Beat/learn:** Bo's Phase 2 uses CE on a static 1 s delay - no explicit latency loss. Adopting this is a free win.

- **Uni-ASR** (arXiv:2603.11123) - in appendix but not adopted. Gap.

- **UAF: Unified Audio Front-end LLM** (arXiv:2604.19221, Apr 2026). One AR head for VAD + turn detection + speaker recognition + ASR + QA. Essentially Bo's architecture, more ambitious. Speaker stream is a plausible v2.

- **Distilling Conversations** (arXiv:2603.26246, Mar 2026). Learned compression of prior-turn audio context to cut KV cost. **Beat/learn:** Bo caps prefix at 2048 tokens via LRU; a learned compressor is a better Phase 4 ablation.

## 2. Semantic / context-aware endpoint detection

- **FastTurn** (arXiv:2604.01897, Apr 2026, ASLP Lab; github.com/ASLP-lab/FastTurn). Fuses streaming CTC acoustic features with LLM hidden states through an "acoustic adapter" before the turn decision. Ships a real-dialogue test set with overlaps, backchannels, pitch variation. **Beat/learn:** Bo's `<END_SPEECH>` is text-stream-only; FastTurn's acoustic+linguistic early fusion directly attacks Bo's Risk #2 (disfluency).

- **Hierarchical End-of-Turn with Primary Speaker Segmentation** (arXiv:2603.13379, Mar 2026, Meta). 1.14M-param MFCC student distilled from wav2vec 2.0. Reports **87.7% recall vs 58.9% for Smart Turn v3, 36 ms median latency vs 800-1300 ms**. **Beat/learn:** the headline number to beat. Bo's 400 ms P50 target looks weak; that paper is 10x faster (smaller model, no transcript).

- **JAL-Turn** (arXiv:2603.26515, Mar 2026). Joint acoustic-linguistic turn-taking, full-duplex. Another fusion baseline.

- **LiveKit turn-detector v0.4.1-intl / MultilingualModel** (HF: livekit/turn-detector, Dec 2025). Qwen2.5-0.5B-Instruct, distilled from a 7B teacher, English + 13 langs, <500 MB CPU. **Beat/learn:** the current LiveKit reference; Bo's plan still names "Smart Turn v2."

## 3. Context biasing / hotword injection

- **LOGIC: Beyond Prompting** (arXiv:2601.15397, Jan 2026; **withdrawn 2026-02-04**, cite cautiously). Trie-based logit-space bias injection; documents prompting failures: "lost-in-the-middle," context-window blowup, over-correction. **Beat/learn:** validates Bo's distractor-prefix intuition; LOGIC is a stronger architectural baseline than concat-into-prefix.

- **Contextual Biasing in Speech LLM with Common Word Cues** (arXiv:2604.12398, Apr 2026). Uses acoustically-similar common-word cues to bias rare-word recognition. Orthogonal to prefix biasing; useful if Bo's approach plateaus on OOV.

- **Hotword Retrieval + GRPO** (arXiv:2512.21828) - in Bo's appendix. GLCLAP retrieval + GRPO. Should be a comparison point at eval.

- **BR-ASR** (arXiv:2505.19179) - in appendix. 200k-entry retrieval; v2 stretch.

## 4. Alternatives to Mimi for ASR

- **FlexiCodec** (arXiv:2510.00981, Microsoft, Oct 2025). **ASR-feature-assisted dual-stream encoder** + Transformer bottlenecks. Inference-time-controllable frame rates **3-12.5 Hz** via adaptive merging of semantically similar frames. **Beat/learn:** most directly relevant codec. 3-6 Hz with semantic preservation cuts KV cost vs Mimi's flat 12.5 Hz. Add as third arm in Phase 0 bake-off.

- **XY-Tokenizer** (arXiv:2506.23325) and **DualCodec** (arXiv:2505.13000). Dual-stream codecs (Whisper or SSL feature + waveform) trained for the same semantic-vs-acoustic tradeoff at low bitrate.

- **Speech Codec Probing from Semantic and Phonetic Perspectives** (arXiv:2603.10371, Mar 2026). Benchmark finding: **ASR-trained features beat SSL features** for semantic preservation; phonetic regularization (SpeechTokenizer/DualCodec/SemantiCodec) helps ASR probes at low rates. **Beat/learn:** Mimi distills WavLM (SSL) - this paper is direct evidence that Mimi tokens may not be ASR-optimal (Bo's Risk #3).

- **AudioCodecBench** (arXiv:2509.02349, Sep 2025). Standardized codec eval.

## 5. Production streaming systems (2026)

- **Deepgram Flux** (2026; multilingual May 2026). First production STT with **model-integrated end-of-turn detection** in the same forward pass as transcription, plus an **EagerEndOfTurn** event firing before silence so the downstream LLM can prefill. Nova-3-level accuracy. **Beat/learn:** validates Bo's one-model bet; Bo should add an explicit "eager" pre-endpoint token rather than only a single `<END_SPEECH>` event.

- **AssemblyAI Universal-3 Pro Streaming** (Mar 2026). Shared backbone for Voice Agent API and streaming STT; prompting, disfluency control, code-switching, diarization, 99+ languages. Slam-1 (explicit speech-encoder-to-LLM adapter) **deprecated**. **Beat/learn:** the Slam-1 deprecation is another data point that adapter-style two-stage stacks are losing to unified streaming models.

- **Kyutai Pocket TTS** (Jan 2026, 100M, CPU realtime); **Hibiki-Zero** (Feb 2026, A3B simultaneous S2ST trained with GRPO without word-level alignment); **MoshiRAG** (Apr 2026, async retrieval for full-duplex). **Beat/learn:** Hibiki-Zero's GRPO-without-alignment recipe is a candidate replacement/companion for Bo's Phase 5 DPO; MoshiRAG is production-style "dynamic context prefix."


---

## Top 3 changes Bo should make

1. **Add FlexiCodec as a third arm in the Phase 0 codec bake-off** (currently Mimi vs Whisper-small). The Mar 2026 codec-probing paper plus FlexiCodec's ASR-feature dual stream directly attack Risk #3. A 3-6 Hz dynamic frame rate also cuts KV cost per session, helping the 500 MB/session target. Sources: arXiv:2510.00981, arXiv:2603.10371.

2. **Re-baseline endpoint targets and add an eager pre-endpoint token.** Meta's hierarchical EOT (arXiv:2603.13379) reports 36 ms median; Deepgram Flux ships EagerEndOfTurn so the downstream LLM prefills before silence. Bo's 400/800 ms targets are weak vs public claims. Add `<EAGER_END_SPEECH>` on high-confidence semantic completion before silence, target <100 ms median, and update baselines from "Smart Turn v2" to LiveKit MultilingualModel + hierarchical EOT + FastTurn + Flux.

3. **Replace the static 1 s text delay in Phase 2 with a learned minimum-latency / monotonic streaming objective.** arXiv:2601.22779 cuts token-emission delay 62.5% with MoChA + a minimum-latency loss; Qwen3-ASR uses dynamic-window attention for the same reason. Train a read/write policy and share weights with a non-streaming twin (Uni-ASR style) so both objectives come for free.
