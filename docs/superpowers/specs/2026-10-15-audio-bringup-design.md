# Audio bring-up: a LoRA or fine-tune of a container-served music model (design)

Date: 2026-10-15. Status: written for the operator's review; nothing is built.
Prompt (operator): "our goal here is to make music happen from a specialized model", then option 1: a real orchard path for audio,
so the run is a ledger run that shows in the web UI. Decisions from the operator: language-model LoRAs first with the merge target a
parameter (answer "yes" to the proposal), and "the harness should self-select the workaround when needed".

## 1. Goal and success

`tt-orchard bringup <adapter-or-finetune> --base <container-bundle>` runs unattended to `ready-for-operator-review` for a music model
whose base is served by a v5.1 container package (first target: `terminusresearch/minimax-music3-lm-lora-fiona-crapple` on
`jashansinghTT/minimax-music3-blackhole`). The review bundle holds clips from the new model and from the base, made with the same
seeds and captions. The run shows in the web UI with players for the clips.

Success for a run means the new model boots in the container, generates audio, and the checks in section 6 pass. It does not mean
the music is good or that the style transferred. That stays a human judgment, and the bundle is built to make it easy: same seeds,
same captions, base next to new.

Proven by hand on 2026-10-15 (outside orchard, see `docs/run-logs/2026-10-music-blackhole-first.md` and the 2026-10-15 entries in
`CLAUDE.md`): the base and a merged copy each booted on one chip and made 20 s clips, and the operator listened and confirmed they
are music.

## 2. Scope

In: LoRA adapters whose keys target one component of the base (language model first), the merge into a model directory, a
container serving check for audio, base resolution for container bases, the bundle, the UI.
Out: stage 7 packaging (a v5.1 package needs a 1.5 to 2.5 hour image build, defaults.py), publishing, adapters that target the
diffusion transformer (the merge target is a parameter, untested), full-weight fine-tunes (they take the same path with
`form: full` and no merge, untested), a quality metric, serving on more than one chip, and fixing the container's hang.

## 3. Three recorded facts, no new class

The model has the base's architecture and only weights differ, so path and class stay `weights-only`. Stage 0 records three more
facts in `delta.json` and in its passing `stage_end`:

| fact | values | meaning |
|---|---|---|
| `form` | `adapter`, `full` | an adapter needs the merge in stage 1 |
| `serving` | `thin-bundle`, `container` | how the base is served |
| `modality` | `text`, `audio` | what the stage 2 check measures |

A reader like `run_path` (`stages.run_class`) replays the facts from the ledger on a resume, so editing `delta.json` later changes
nothing. A ledger with none of them means `full`, `thin-bundle`, `text`: every existing run keeps its stage table.
`spec_for(number, path, package_format, cls, facts)` returns the audio variants only for `serving: container` with
`modality: audio`.

## 4. Stage flow for an audio run

| stage | what happens |
|---|---|
| 0 triage | decide the facts (section 5) |
| 1 materialize | merge the adapter into `model-dir/` (section 5); no hardware |
| 2 boot and generate | one lease, a probe, then two container boots, base and new (section 6) |
| 3, 4, 5, 6 | skipped, each with a stated reason (the container profile is one chip; no quality check exists) |
| 7 | skipped: `--package-format v5.1` is refused at start (the image build); v6 does not apply to a container base |
| 8 | the bundle, with clips (section 8) |

Skip reasons are written the way `SKIP_3_WEIGHTS_ONLY` and the others are, as constants in `stages.py`.

## 5. Base resolution, triage and materialize

Base resolution. `nearest.read_installed` also reads container manifests under `<tt-model cache>/pulled/*/tt_kernel_manifest.json`
(schema 5.1, `container.kind`, `weights.repo_id`, `device_count`) and `Installed` gains `serving` and `kind`. `cli.resolve_base`
accepts them. The preflight base check says, before any pull, whether the base is usable. The old behaviour for a thin bundle does not
change.

Adapter recognition. The preflight base check treats a model repo with `pytorch_lora_weights.safetensors` or `adapter_config.json` as an
adapter. Its base comes from the card's `base_model` or `--base`, as for any model.

Stage 0 triage template (`delta_triage.py`) gains an adapter branch. It reads the adapter header only, with the same 8 MiB bound the
research code uses, and checks:
- every tensor is `<prefix><name>.lora_A.weight` or `.lora_B.weight` with matching ranks, one prefix, and every adapted matrix has the
  base tensor `<name>.weight` with shape `(out, in)` equal to `B @ A`;
- the component is the base sub-directory whose weight index holds those names (`language_model` here);
- alpha has a stated source, tried in this order: `adapter_config.json` (`lora_alpha`), the safetensors metadata, then the card's text
  (`alpha 64` or `lora_alpha=64`, accepted only when every match gives the same number). The fiona adapter has no config and no
  metadata; its card says "rank 64, alpha 64" and shows `lora_alpha=64`, so it resolves from the card. The source is recorded. When
  no source gives a number, or two give different numbers, triage writes `alpha: unknown`, the class is `unknown` and the run blocks
  as `adapter-alpha-unknown`. `--adapter-alpha N` overrides every source and is recorded in the ledger. The harness never guesses.

Stage 1 materialize template (`materialize_adapter.py`, a wrapper around `merge_lora.py`) reads `merge_config.json` (base snapshot,
adapter file, component, prefix, alpha, strength 1.0) and writes `model-dir/` and `MERGE.json`. It refuses when the merged count
differs from the adapter's module count. An agent writes the config and runs the template; it never edits weights.

## 6. Stage 2: boot and generate

Template `serve_and_generate.py` (container), run by the supervisor's single-test hardware phase under its lease. It:
- maps each leased chip to its device node (`/sys/bus/pci/devices/<bdf>/tenstorrent/`) and passes only those nodes, each under its
  own number (`/dev/tenstorrent/2:/dev/tenstorrent/2`). Renumbering a chip to 0 is wrong: UMD's `CHIP_IN_USE_<n>_PCIe` lock is a file in
  `/dev/shm`, which `--ipc host` shares between containers, so two containers on different boards that both call their chip 0 wait
  on one lock (measured 2026-10-15: a second container blocked at "Waiting for lock 'CHIP_IN_USE_0_PCIe'"; with the nodes kept under their
  own numbers both ran at once). `tt-model serve --print` pins whichever node it likes, so the template rewrites it and requires exactly one
  `--device` argument;
- creates the cache directories before `docker run` (a missing mount source is created by Docker as root);
- mounts the model directory, and every directory its links point into, read-only at their own paths, and sets `HF_MODEL`;
- has a boot deadline and a stall rule (no new container log line for a set time) and saves the container log before it removes
  the container. This cleanup is best effort only. On a timeout or an abort the supervisor kills the test's process group with
  SIGKILL (`orchard.agent._kill_session`), which skips a `finally`, and killing the Docker client does not stop its container
  (an 8-hour hang on 2026-10-15 left one running). The guarantee is the supervisor's, next paragraph;
- starts every container it makes, the probe's and the server's, with the run label the supervisor passes to hardware tests
  (`ORCHARD_TEST_LABEL`), and drops the tt-model labels, as stage 4's container template does: those labels make a container look
  like a tt-model server to `tt-model stop` and to tt-model's free-chip scan.

Supervisor-owned cleanup (a finding of the PR 8 review). Stage 2's single-test hardware phase (`_hardware_phase`) has no container
sweep today; only the listed-tests path calls `_sweep`. The change for this design: after every stage 2 hardware test, whatever its
outcome (passed, failed, timed out, aborted), and again at resume when recovery finds a test in progress, the supervisor lists
containers by the run label and stops and removes each one. If any is still listed afterwards the stage blocks, as `_sweep`
does now, and no lease is released or reset and the coder is not restored until a person looks. The sweep also runs before the coder is restored.

Hang probe and the workaround (the operator's ruling: the harness self-selects it). Before the first boot the template runs a probe
in the same image on the leased chip: `ttnn.as_tensor` twice with one `cache_file_name` for a (2, 8, 16384, 128) bf16 tile tensor, a
64 MiB flatbuffer, under a 60 s timeout. On this host (Linux 7.0.0-38, KMD 2.11.0, firmware 19.16.0, image tt-metal `v0.79.0-dev20260907`)
the load hangs; on a host without the problem it takes under a second.
- Probe passes: boot as is.
- Probe hangs: the template kills the probe container, records `workaround selected` with the probe's evidence, and boots with a copy
  of the image's `ttnn/operations/core.py` in which exactly one line is changed so `as_tensor` never uses the flatbuffer cache. The
  edit is asserted (the line must occur exactly once), the file is mounted read-only, and its sha256 and the edit go in
  `result.json`, the ledger and the bundle's risks. It changes vendor code in a mount and only for this container.
- The probe is one data point about the hang. If a boot without the workaround still stalls after a passing probe, the run
  blocks as `container-cache-hang` with the stall point and the log tail, and does not retry on its own.
- To measure before building: whether the chip works after a killed hung probe inside the same lease (if not, the template asks
  the supervisor for a `gozer reset` of the lease, or exits with a code that makes the supervisor rerun the test after the release
  reset).

Generation. For each of two captions (a descriptive one with the adapter's trigger when its card names one, and one without) and two
seeds, on each of the two models, the template requests a clip of 20 s through the job API (`/v1/music/jobs`), saves FLAC and records
the job's stage times and the real-time factor. The base boots first.

Checks (function only; the bars are choices, set from 24 real clips measured on 2026-10-15: duration 20.02 s for 20 s asked, 32 kHz
stereo, at most 0.019% of samples at full scale, at least 59% of 100 ms windows above -40 dBFS):
- duration within 5% of the request, sample rate and channel count equal to the base's;
- at least 40% of 100 ms windows above -40 dBFS, and at most 1% of samples at full scale;
- a repeated request on the new model is bit-identical;
- for each caption and seed, the new model's file differs from the base's.
`gate_audio_generate` reads `result.json` and checks every number again from the saved files, as `gate_package` does. A failure is a
failed stage with the failing check named.

## 7. Failure and resume

The stage uses the single-test hardware phase and its `test-result.json` resume marker. A failed test gets no gate feedback (existing rule).
A hang ends the test at the deadline with `timed_out`; the supervisor kills the process group and then sweeps by label (section 6),
so cleanup does not depend on the template's own `finally`. A stale cache from an earlier attempt is moved aside by the existing rule. New block codes: `container-cache-hang` and
`adapter-alpha-unknown`, each with a "how to unblock" text and a test that ties them to `blocked.REASONS` and `narrate.UNBLOCK`.

## 8. Bundle and UI

Stage 8 (`build_bundle.py`) adds `audio/` (the FLAC clips, `features.json`, `MERGE.json`), a table of base and new clips with seeds and
captions, and risks: "audio quality is not judged", "vendor patch used: yes or no (sha256)", "alpha source". Hazard rows come from the
evidence, as the existing ones do. Clips are copied only when each is under a size cap (20 MB).

Web UI. `ALLOWED` gains `stages/2/audio/*.flac` and the bundle's `audio/*.flac`; the response type is `audio/flac`, with the existing
no-symlink, size and Host rules. The Files tab shows an audio player for each FLAC. A short card on the run lists base and new clips side
by side per caption and seed, with the checks. All server strings are set with `textContent`, as now. The run's facts (`form`,
`serving`, `modality`) show in the Details tab. The tests for the page's JavaScript are manual and by `node --check`, as before.

## 9. Testing

Tests are written first, one mutation per guard, each seen red. A stdlib fake container server implements the job API and returns
synthetic audio (a tone with a controlled loudness and clipping), so no test needs a chip. Key tests: container manifests are read; a
thin-bundle run is unchanged; triage on a synthetic adapter (rank mismatch, a missing base tensor, alpha unknown, wrong component);
materialize refuses a count mismatch; the template rewrites the device argument and refuses two; the probe hang selects the workaround and a passing probe does not; the work-around edit is asserted exactly once; each audio check fails when its input is wrong;
a repeat that differs fails; the gate re-derives from files; the run is killed and resumed at stage 2 boundaries (the existing
kill-point test pattern); a fake containers adapter shows the label sweep after a timeout, after an abort and after a kill and resume of
a stage 2 test, and that a surviving container blocks before any release, reset or coder restore; the UI serves a FLAC only from the allowed paths. Then a real run under the supervisor on
`fiona-crapple`, killed once and resumed. The existing suite must stay green.

## 10. Open points

- Whether a chip is usable after a killed probe in the same lease (section 6): measured first, before the plan fixes the step.
- The boot deadline and the stall time: choices until a cold boot's longest silent stretch has been measured from container logs
  (cold boots here took 76 to 165 s with the workaround; the 8-hour hang had no output at all).
- The probe covers one tensor size. It does not prove other cache paths are safe.
- A second audio base: nothing else is known to exist as a TT container for music, so the code is written for one kind
  (`tt-dit-server`) and the manifest's `container.kind` is checked.
- The listening step is still the operator's.
