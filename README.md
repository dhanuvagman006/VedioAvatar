# VedioAvatar

Upload a short clip of a person talking to camera (face visible, their voice on the audio
track) plus a script, and get back a video of **the same person, in their own voice, reading
your script**.

```
   input clip (face + voice)                 script.txt
            │                                    │
            ├─► reference voice (12 s) ──► Chatterbox voice-clone TTS ──► speech.wav
            │                                                                │
            └─► silent 25 fps driver ──► loop to speech length ──► Wav2Lip / LatentSync lip-sync
                                                                             │
                                                            ffmpeg mux ──► output.mp4
```

Everything runs locally on one NVIDIA GPU. It is sized for an **RTX A2000 Laptop (4 GB
VRAM)** running Windows, but works the same on Linux / WSL2 and on bigger GPUs (where it can
switch to the higher-quality LatentSync model automatically).

## Status

| Part | State |
|------|-------|
| Orchestration, ffmpeg preprocessing, ping-pong looping, job queue, REST API, web UI, CLI | Implemented and covered by the test suite (runs with mock TTS / lip-sync backends, no GPU needed). |
| Chatterbox TTS worker, Wav2Lip worker, LatentSync worker | Implemented against the upstream projects' current command-line contracts. They need a first run on the GPU machine; see *Troubleshooting* if something in the upstream stack has drifted. |

## Hardware notes (RTX A2000 4 GB)

The stages run one after another, each in its own process, so VRAM is released between them.

| Stage | Model | VRAM | Notes |
|-------|-------|------|-------|
| Voice clone TTS | Chatterbox **Turbo** (350M, default) or **Nano** (110M) | ~2–3 GB | English. Nano also runs fine on CPU. |
| Voice clone TTS | Chatterbox English (500M) / Multilingual (23 languages) | ~4 GB+ | On a 4 GB card these usually fall back to CPU automatically (slower, still works). |
| Lip-sync | **Wav2Lip** (default when VRAM < 9 GB) | ~1.5–2.5 GB | Robust, fast, mouth region is soft (96 px model). |
| Lip-sync | LatentSync 1.5 (optional) | ~8 GB | Much sharper. Not usable on 4 GB; picked automatically on >= 9 GB GPUs. |

Rough wall-clock on the A2000 for the bundled 20 s clip and a 35 s script: 1–3 minutes after
the first run (the first run also downloads ~2 GB of Chatterbox weights).

## Setup on Windows (the GPU laptop)

Prerequisites:

1. **Python 3.11** from <https://www.python.org/downloads/windows/>. In the installer tick
   **Add python.exe to PATH** (and keep *py launcher* ticked). Or: `winget install -e --id Python.Python.3.11`.
   A fresh Windows install has a fake `python` that only prints *"Python was not found; run without
   arguments to install from the Microsoft Store"*: that is not Python, install the real one and
   open a **new** terminal afterwards.
2. **Git** from <https://git-scm.com/downloads>.
3. A current NVIDIA driver (the one already installed is fine).
4. About 12 GB of free disk and a decent connection (PyTorch CUDA wheels are ~2.5 GB each and
   are installed into two separate environments).

Then, in **cmd or PowerShell**:

```bat
git clone https://github.com/dhanuvagman006/VedioAvatar.git
cd VedioAvatar
powershell -ExecutionPolicy Bypass -File setup.ps1
```

`setup.ps1` creates `.venv` (the lightweight orchestrator), then runs
`scripts/setup_envs.py`, which:

* creates `envs\tts` (Python 3.11, torch 2.6.0 CUDA 12.4, `chatterbox-tts`),
* creates `envs\wav2lip` (Python 3.10, torch 2.6.0, librosa, opencv) and clones + patches
  [Wav2Lip](https://github.com/Rudrabha/Wav2Lip) into `third_party\Wav2Lip`,
* downloads the Wav2Lip weights (`wav2lip_gan.pth`, `s3fd.pth`, ~530 MB) into `models\`,
* prints a `doctor` report.

The extra Python versions are fetched automatically by [uv](https://github.com/astral-sh/uv);
you do not have to install them yourself. ffmpeg is optional: if it is not on `PATH` the
`static-ffmpeg` package downloads a build on first use.

Everyday use afterwards, from the repo folder. `avatar.bat` runs the CLI with the project's
virtual environment, so nothing needs activating (in PowerShell type `.\avatar` instead of `avatar`):

```bat
avatar doctor        # all rows except latentsync should say OK
avatar run --video examples\sample_face.mp4 --script examples\sample_script.txt --out out.mp4
avatar serve         # web UI + API on http://localhost:8000
```

If you prefer an activated shell: `.venv\Scripts\activate.bat` (cmd) or
`.\.venv\Scripts\Activate.ps1` (PowerShell), then `python -m avatar_pipeline.cli ...`.

## Setup on Linux / WSL2

```bash
sudo apt install -y git ffmpeg libgl1        # libgl1 is needed by opencv
git clone https://github.com/dhanuvagman006/VedioAvatar.git && cd VedioAvatar
./setup.sh                                   # add "--lipsync both" on a >= 8 GB GPU
./avatar doctor                              # ./avatar wraps the CLI; or: source .venv/bin/activate
```

WSL2 uses the Windows NVIDIA driver directly; no driver install inside WSL is needed.

## Usage

### Command line

```bash
python -m avatar_pipeline.cli run \
  --video my_clip.mp4 --script my_script.txt --out result.mp4 \
  --tts-model turbo --lipsync-backend auto
```

Useful flags (all optional, defaults come from `config.yaml`):

| Flag | Meaning |
|------|---------|
| `--text "..."` | Give the script inline instead of a file. |
| `--ref-audio voice.wav` | Clone the voice from this file instead of the clip's own audio track. |
| `--tts-model turbo\|nano\|english\|multilingual` | Chatterbox variant. Use `multilingual` with `--language hi`, `de`, `fr`, ... |
| `--tts-device auto\|cuda\|cpu` | `auto` retries on CPU after a CUDA out-of-memory. |
| `--lipsync-backend auto\|wav2lip\|latentsync\|mock` | `auto` chooses by VRAM. |
| `--exaggeration 0.7 --cfg-weight 0.3` | More expressive delivery (english / multilingual models). |
| `--denoise` | Light denoise of the voice reference when the recording is noisy. |
| `--workdir DIR` | Keep the intermediate files (reference.wav, speech.wav, driver.mp4, manifest.json) here. |

`python -m avatar_pipeline.cli chunks --script my_script.txt` shows how the script is split
into TTS chunks.

### Web UI

`python -m avatar_pipeline.cli serve` then open <http://localhost:8000>: pick the clip, paste the
script, press *Generate*. The page shows stage progress, the live log, and the finished video.
To reach it from another machine on the LAN use its IP (the server binds `0.0.0.0:8000`).

### REST API

```bash
# submit (multipart form). Returns 202 + the job record
curl -F video=@examples/sample_face.mp4 -F script_file=@examples/sample_script.txt \
     -F tts_model=turbo -F lipsync_backend=auto http://localhost:8000/api/jobs

# poll
curl http://localhost:8000/api/jobs/<id>            # {"status":"running","stage":"tts","progress":41.2,...}
curl http://localhost:8000/api/jobs/<id>/log?tail=50
# download when status == "done"
curl -o result.mp4 http://localhost:8000/api/jobs/<id>/result
```

Form fields: `video` (required), `script` or `script_file`, `ref_audio`, `tts_backend`,
`tts_model`, `language`, `lipsync_backend`, `exaggeration`, `cfg_weight`, `temperature`,
`denoise`. Other endpoints: `GET /api/health`, `GET /api/jobs`, `DELETE /api/jobs/{id}`.
Jobs run one at a time (single GPU) and survive a server restart in the queue.

## Configuration

Defaults live in [`config.yaml`](config.yaml) (`AVATAR_CONFIG=path` or `--config` overrides
the file). The interesting knobs:

| Key | Default | Effect |
|-----|---------|--------|
| `preprocess.reference_seconds` | 12 | How much of the clip's speech is used as the voice reference (10–15 s is the sweet spot). |
| `preprocess.max_height` | 720 | Driver video is downscaled to this; lower = faster lip-sync, less VRAM. |
| `preprocess.max_source_seconds` | 60 | Longer clips are trimmed before looping. |
| `tts.model` | `turbo` | Chatterbox variant (see hardware table). |
| `tts.max_chunk_chars` | 250 | Script is synthesised in sentence chunks of this size. |
| `lipsync.backend` | `auto` | `auto` = LatentSync when VRAM >= `auto_latentsync_min_vram_mb`, else Wav2Lip. |
| `lipsync.wav2lip.pads` | `[0,10,0,0]` | Extra pixels below the chin; raise to 15–20 if the chin gets cut. |
| `lipsync.wav2lip.resize_factor` | 1 | Set 2 on very tight VRAM or for speed. |
| `output.crf` | 18 | Final H.264 quality (lower = better/larger). |

## How the pipeline works

1. **probe** — ffprobe validates the clip (video stream, >= 2 s, audio present unless `ref_audio`).
2. **reference** — extract mono 24 kHz audio, strip leading silence, loudness-normalise, keep 12 s.
3. **driver** — silent copy of the clip at 25 fps, <= 720 px tall, even dimensions.
4. **tts** — `workers/tts_chatterbox_worker.py` (inside `envs/tts`) splits the script into
   sentence chunks, generates each in the cloned voice, joins them with 0.3 s pauses.
5. **extend** — the driver is looped forward/backward (ping-pong, no jump cuts) or trimmed so its
   length matches the speech.
6. **lipsync** — `workers/wav2lip_worker.py` or `workers/latentsync_worker.py` re-renders the
   mouth region frame by frame to match the speech.
7. **finalize** — re-encode to H.264 + AAC, `+faststart`, exact speech length.

Every stage writes into the job's `work/` folder and `manifest.json` records timings, backends
and input properties.

### Why three virtual environments?

Chatterbox pins `torch==2.6.0` / `transformers==5.x`, LatentSync pins `torch==2.5.1` /
`transformers==4.48`, and Wav2Lip is a 2020 code base that needs small patches. Putting them in
one environment does not resolve. The orchestrator (`avatar_pipeline/`) has only light
dependencies and launches each model as a subprocess in its own interpreter
(`avatar_pipeline/backends/runner.py`), streaming logs and `@@PROGRESS` lines back.

## Recording tips (this matters more than any setting)

* 15–40 s, one person, face fully visible the whole time, roughly frontal, no hands over the mouth.
* Speak naturally and continuously; the first 12 s of speech become the voice reference.
* Quiet room, no music, no other voices. Phone mic at arm's length is fine.
* Good, even lighting; avoid strong backlight and heavy motion blur.
* The mouth should not be tiny in frame: a head-and-shoulders portrait crop works best.

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `Interpreter not found: envs/...` | Run `python scripts/setup_envs.py` (or `setup.ps1`). |
| `CUDA out of memory` in the TTS stage | Use `--tts-model nano`, or `--tts-device cpu`. `auto` already retries on CPU. |
| Wav2Lip: *could not find a face in at least one frame* | Re-record with the face visible throughout, or trim the clip. `temp/faulty_frame.jpg` inside `third_party/Wav2Lip` shows the offending frame. |
| Wav2Lip OOM during face detection | Lower `lipsync.wav2lip.face_det_batch_size` (it also auto-halves on OOM) or set `resize_factor: 2`. |
| Voice sounds off / robotic | Give a cleaner reference (`--ref-audio` with a studio-ish recording), try `--tts-model english --exaggeration 0.4 --cfg-weight 0.4`, or `--denoise`. |
| Mouth looks blurry | That is Wav2Lip's 96 px limit. Use a tighter head-and-shoulders crop so the face is larger in frame, or run LatentSync on a >= 8 GB GPU. |
| `chatterbox` import errors after an upstream release | Pin a version: `uv pip install --python envs/tts/bin/python chatterbox-tts==<version>` and open an issue. |
| ffmpeg not found | Install ffmpeg or `pip install static-ffmpeg` in `.venv`; set `FFMPEG_BIN` / `FFPROBE_BIN` if it lives elsewhere. |
| LatentSync on Windows fails to install | Use WSL2 (Ubuntu) for LatentSync; Wav2Lip and Chatterbox work natively. |

`python -m avatar_pipeline.cli doctor` checks ffmpeg, the GPU, every environment (imports torch
and reports CUDA availability) and the model weights.

## Project layout

```
avatar_pipeline/        orchestrator package (light deps)
  pipeline.py           the 7-stage pipeline
  media.py              ffmpeg / ffprobe helpers (probe, reference audio, ping-pong loop, mux)
  text.py               script normalisation + sentence chunking
  backends/             TTS / lip-sync adapters (chatterbox, wav2lip, latentsync, mock)
  jobs.py               file-backed job store + single background worker
  server.py             FastAPI app (REST + serves web/index.html)
  cli.py                `avatar run | serve | doctor | chunks`
workers/                scripts executed inside the per-model environments
scripts/setup_envs.py   creates envs/, clones + patches upstream repos, downloads weights
scripts/download_models.py, scripts/patch_wav2lip.py
web/index.html          single-page upload / progress / preview UI
examples/               20 s sample clip + sample script
tests/                  pytest suite (mock backends, needs ffmpeg only)
```

Run the tests with `pip install -r requirements-dev.txt && python -m pytest`.

## Limitations and next steps

* One job at a time; no cancellation yet (delete the job after it finishes).
* Wav2Lip output quality is limited by its 96 px mouth model. A face-restoration pass
  (GFPGAN / CodeFormer on the mouth crop) would be the next quality win that still fits in 4 GB.
* Chatterbox Turbo/Nano are English-only; use `multilingual` for other languages (larger model).
* Head motion in the output is whatever the source clip contained (looped). Very long scripts
  therefore look repetitive; 30–60 s of source footage is ideal.
* Do not use this on someone's face or voice without their consent.

## Licenses

This repository is MIT. It downloads and drives third-party projects with their own licenses:
[Chatterbox](https://github.com/resemble-ai/chatterbox) (MIT),
[Wav2Lip](https://github.com/Rudrabha/Wav2Lip) (code: see upstream; weights for research /
personal use per the authors), [LatentSync](https://github.com/bytedance/LatentSync)
(Apache-2.0). `examples/sample_face.mp4` is a LatentSync demo asset (Apache-2.0).
