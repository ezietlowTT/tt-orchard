# Music generation on Blackhole: what the current port looks like, and what a Blackhole-first model would change

Written 2026-10-14 during the Sonexa music run. The first part comes from reading the `tt-model/minimax-music3` container
(manifest, env settings, `config.py`). The second part is design reasoning. Nothing here is measured on a chip.

## The current model (MiniMax-Music3 and its TT bundle)

- The model is a pipeline: a Qwen3-style language model (17 GB, 4 shards), an RVQ depth decoder (1.3 GB), a diffusion
  transformer (9.8 GB), a condition encoder and a vocoder. A 9.8 GB VAE and `dav.pth` are left out of the TT bundle.
- The bundle (`jashansinghTT/minimax-music3-blackhole`, a v5.1 container package, `tt-dit-server`) runs all of it on one p150
  chip with no fabric (`MUSIC3_FABRIC_CONFIG=none`). It fixes a maximum of 16,384 tokens and 9,000 frames, warms up at 200
  frames and reserves a 256 MB trace region.
- The server reads a model directory from `MUSIC3_SNAPSHOT`, or a directory in `HF_MODEL`, before it asks the Hub.
- It was designed for GPUs and ported. The port keeps the stages as they are and fits them on one chip.

## What a Blackhole-first model would change

1. **Different parts on different chips.** Language-model decode is limited by memory bandwidth. The diffusion transformer is
   limited by matrix-multiply speed over a long sequence. The vocoder is mostly convolution. A QuietBox 2 has four chips joined
   by Ethernet, so a design could pipeline them (tokens, latents, audio). The current bundle uses one and leaves three idle.
   One clip per chip raises throughput with no fabric.
2. **Fixed shapes from the start.** TT traces work best when shapes do not change. Generating in fixed-size overlapping chunks
   lets every step reuse one compiled trace, and gives streaming output. Padding a variable-length clip up to a maximum does
   neither.
3. **Few diffusion steps.** A distilled or consistency-trained transformer runs 4 to 8 steps. Conditioning is computed once and
   reused, and the step loop sits in one trace.
4. **Fewer small sequential steps in the depth decoder.** One-at-a-time RVQ codebook prediction is many tiny kernels, which TT
   launches slowly. Predicting several codebooks per pass cuts the count.
5. **Ops chosen for the hardware.** Large 1D convolutions, STFT and resampling are awkward on TT (the current server resamples
   on the host with `soxr`). A vocoder built from matmuls (patchified or transformer layers) runs on the same cores as the rest.
6. **Precision chosen by listening.** Coder-Next held up with bfp4 experts on text. Audio is likely more sensitive and has no
   token-agreement test, so each choice needs a listening panel or an audio metric.

## What it means for orchard

- Top-1 agreement with a CPU reference does not apply to audio. The sidecar parity check is closer: run one seed on CPU and chip
  and compare latents by correlation (PCC).
- The speed number is real-time factor (seconds of audio per second of compute), not tokens per second.
- The minimum check is boot, request a clip, and get a non-silent FLAC of the expected length. A seeded latent comparison is the
  next step up.

## Result of the hand run (2026-10-15)

Measured results and the limits of the comparison are in `~/orchard-runs/music/RESULTS.md` (not committed: it names clips that live in the
run directory). Short version: the base and a LoRA-merged copy of MiniMax-Music3 each booted and made a 20 s clip on one chip, at
about 3.2 s of compute per second of audio, once a host problem was worked around (the image's tensor-cache load hangs above
about 32 MiB on this box). Stage times for the 65 s run (20 s of audio): diffusion denoise 23.9 s, depth decoder 19.1 s, language model 18.0 s, vocoder 14.6 s,
VAE decode 3.5 s. Denoise is the largest, which supports point 3 (few diffusion steps). The language model (36 ms per frame) and the depth
decoder (38 ms per frame) cost about the same, so point 4 (fewer sequential codebook steps) matters as much as faster language-model decode.
The vocoder takes 22% of the time, which supports point 5. These are one run each.
