# Example inputs

| File | What it is |
|------|------------|
| `sample_face.mp4` | 20 s, 1080x1920, 25 fps clip of a person talking to camera, with their voice on the audio track. |
| `sample_script.txt` | ~85 words (about 35 s of speech) for the person to "read". |

Run the whole pipeline on them:

```bash
python -m avatar_pipeline.cli run --video examples/sample_face.mp4 --script examples/sample_script.txt --out out.mp4
```

The clip is shorter than the script, so the pipeline loops it forward/backward (ping-pong)
until it covers the generated speech, then re-renders the mouth to match.

## Attribution

`sample_face.mp4` is `assets/demo2_video.mp4` from the
[ByteDance LatentSync](https://github.com/bytedance/LatentSync) repository, redistributed
unchanged under its Apache License 2.0. It is included only as a convenient test input.
Replace it with your own recording for real use (see the recording tips in the main README).
