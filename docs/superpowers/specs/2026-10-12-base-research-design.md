# Researching the base model for a bring-up

Date: 2026-10-12. Status: design, not built. Branch `research-base-discovery`.

## 1. Intent

Operator prompt (2026-10-12): "How do we make this project better at finding and researching the model
family it might need to base a new bring up on? Searching tt model manager, inspecting existing models,
using HF or GH.. not just looking locally." Scope chosen: everything (lineage, architecture match,
class estimate before download, search for existing ports, a written report).

Success: before a run downloads anything large, the operator and stage 0 can see which models the new
model descends from, which bundle serves the closest one, what class of work to expect, and whether
someone has already started a port. The humanizer run (a Gemma 4 fine-tune, 3 coder boots and 105 agent
turns spent on `ls`) must end with a ranked answer instead of a block that lists names.

Assumptions (the operator did not state these):
- Hub and GitHub calls are acceptable from the machine that runs `bringup`.
- The class estimate is advisory. Stage 0's own measurement decides the path, as today.

## 2. What exists

`orchard/nearest.py` reads installed bundles and runs `tt model search <family> --json`.
`preflight.check_base` takes the card's first `base_model` (or `--base`), finds an installed bundle whose
manifest `weights.repo_id` equals it, and otherwise blocks with `nearest-model-missing`, listing bundles
whose weights repo shares the leading letters of the base's name. Gaps:
- One signal. A card with no `base_model` gives a warning and nothing else. `base_model_relation` is not read.
- "Nearest" means the same repo name. No architecture comparison exists.
- Nothing looks at the hub's other models, tt-metal, the model catalog or GitHub.
- A 50 GB download happens before anyone knows whether the model is a weights swap or a full port.

Checked on 2026-10-12 from the lab box:
- `tt model search <q> --json` returns `{"bundles": [{name, private, downloads, last_modified, installed}]}`.
  `--catalog` limits it to the community catalog and `--arch` filters by chip family.
- `GET huggingface.co/api/models?filter=base_model:finetune:<id>` works with no token and lists sibling
  fine-tunes. A model's tags carry `base_model:<relation>:<id>`.
- `GET api.github.com/search/code` returns 401 without a login. Issue and PR search, directory listings
  and raw files work without one (60 requests an hour). `gh` is logged in on this box.

## 3. Design

New package `orchard/research/`, standard library only. One module per source. Each returns a list of
`Evidence(source, claim, url, confidence, fetched_at)` or `Unavailable(source, reason)`. The scorer and
the report contain all decisions. No model takes part.

| Module | Reads | Produces |
|---|---|---|
| `lineage.py` | hub model API: card `base_model`, `base_model_relation`, tags; reverse query for siblings | the chain from the new model up to its root, and sibling fine-tunes. With no card base, an inferred base from `config.json` (`model_type`, `architectures`, layer count, hidden size) |
| `arch_match.py` | `config.json` of each installed or published bundle's weights repo | per candidate: matching and differing config fields |
| `headers.py` | safetensors headers by HTTP range request | tensor names, shapes and dtypes for the new model and a candidate, so the class estimate and the download size exist before any weight is fetched |
| `ports.py` | `tt model search --catalog`; tt-metal `models/` listing and supported-models list (raw files); GitHub issue and PR search for the architecture class name; code search when `gh` is logged in | existing or partial ports, each with a link |
| `report.py` | all of the above | `base-candidates.json` and `base-candidates.md` in the run directory |

Published bundles carry `tt_kernel_manifest.json` at the root (its `weights.repo_id` says which model they
serve) and tags (`thin`: v6; `tt-model-container`: 5.1). A container bundle that lists
`code/models/autoports/<model>/` ships a port; `ports.py` reports it as evidence.

`headers.py` imports the comparison rules from `orchard/skills/delta-triage-templates/delta_triage.py`
(the shared-tensor shape and dtype rule, the sidecar rule, the `mtp.*` allowance). The rules are not copied.
A test fails if the two disagree on a fixture.

### 3.1 Confidence

- `stated`: the card or `--base` names it.
- `inferred`: the card names none, and the architecture fields and tensor shapes match.
- `weak`: only a name or tag overlap.

Stage 0 treats `stated` and `inferred` as comparison targets. A `weak` candidate appears in the report
and is never selected by itself.

### 3.2 Candidate ranking

A candidate is a model that stage 0 could compare against and a bundle could serve. Order: an installed
v6 thin bundle serving the candidate, then a published bundle, then an installed container package, then
a model with no bundle. Within a group: higher confidence first, then fewer differing config fields,
then the cheaper expected class (`weights-only`, `weights+sidecar`, `full-port`). Each row says why it is
where it is.

### 3.3 Preflight integration

`check_base` calls the report. It still blocks with `nearest-model-missing` only when no installed or
published bundle can serve any candidate. The block now carries the ranked list with the expected class
and an existing-port link for each, and the `--base` prompt chooses from that list. When a candidate has
an installed v6 bundle, the check is OK and its message names the expected class and download size. The
snapshot download is unchanged.

`bringup` hands `base-candidates.json` to the supervisor as `--input base_candidates=<path>`. The
delta-triage skill reads it with the existing `base` input and records, in `delta.json`, which candidate
it used and whether the measurement agreed with the estimate.

### 3.4 Flags

`--offline` skips every network source and the report says so. A dry run makes the same read-only research calls as a real run, because it is how the operator reads the
ranked list. It keeps no cache and writes no file. `--offline` makes no network call. `--no-github` skips GitHub only.

## 4. Rules

- Every source is optional. A timeout, a rate limit, a 401 or no network yields `Unavailable(reason)` in the
  report. An unavailable source is never shown as an empty result.
- Hub calls carry no token. The existing rule that `bringup` never uses the operator's HF token holds.
  `gh` is used only for read-only code search. Missing or logged out, the report says that search was skipped.
- Fetched text is untrusted data. It is length-limited, passed through `orchard/scrub.py`, stored as
  evidence, and never placed in a command line or in a prompt as an instruction.
- Responses are cached in the run directory with the fetch time. A resume reads the cache, so the report
  does not change between attempts. `--refresh-research` ignores the cache.
- The collectors write only inside the run directory. They never publish, push, pull or install.
- Hosts are fixed: `huggingface.co`, `api.github.com`, `raw.githubusercontent.com`. Hugging Face answers a
  file request with a signed redirect to a CDN host, so a redirect is followed only when its host ends in
  `.hf.co` or `.huggingface.co`. Any other host, and any `http://` URL, is refused.
- Header reads fetch only the first `N` bytes the safetensors format names (the length prefix, then the
  header), with a cap of 100 MB per file.

## 5. Testing

Written first. Each collector runs against recorded real responses kept in `tests/fixtures/research/`
(the Gemma search above, the reverse query, a GitHub 401, a rate-limit reply, a partial header). No test
uses the network.

Mutations that must each turn a test red, with the suite green first:
no token on hub calls (a token in the environment must not appear in a request), untrusted text not
scrubbed, an unavailable source reported as an empty result, a `weak` candidate selected, the cache ignored
on resume, a redirect to another host followed, a header read with no size cap, the class rules copied and
changed, `--offline` still calling the network, a dry run making calls.

One opt-in live test (like the `~/.qwen` replay) runs the humanizer case: `jialinyyzz/humanizer` must rank
the installed or published Gemma bundles and report the Gemma 3 and Gemma 4 relationship.

## 6. Not in this design

- No model agent does research. A later change could let the stage 0 agent read the report; it does not
  have to search.
- No write access to GitHub or the hub.
- The class estimate does not replace stage 0, and a wrong estimate costs a note in `delta.json`, not a run.
- The class estimate says weights-only, full-port or unknown, and flags sidecar_possible. It cannot see a sidecar's tensors without downloading them.
- Models with no safetensors (GGUF only, pickle weights) get `Unavailable("no safetensors")` for the
  class estimate.

## 7. Open points

- Rate limits: unauthenticated GitHub allows 60 requests an hour. The report uses about 6 per run, with
  the cache. A shared lab box running many bring-ups could exhaust it. Using `gh`'s token for issue and PR
  search too would raise the limit; the design leaves that off until the operator says otherwise.
- The tt-metal supported-models list has no stable machine-readable form that I have verified. The first
  task reads the file and writes a parser with a fixture; if the format is unstable the module reports
  `Unavailable` and the directory listing alone is used.
