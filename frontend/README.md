# A.U.D.I.O — AI-Driven Uncertainty and Detection for Impersonation Overwatch

## Federated Voice Threat Intelligence Platform

React, TypeScript, Vite, and Tailwind CSS frontend for the voice threat intelligence project. File inference connects to the local FastAPI service. History persistence, chat, and live streaming inference require separate endpoints and are not fabricated by the interface.

## Run on Windows

From this directory, in PowerShell:

```powershell
npm install
Copy-Item .env.example .env.local
# Edit .env.local if the FastAPI service is not at http://127.0.0.1:8000
npm run dev
```

Run `npm run dev -- --host 127.0.0.1 --port 5174 --strictPort` and open `http://127.0.0.1:5174`. This matches the configured local FastAPI CORS origin. Use `npm run build` for a production build, `npm run lint` for ESLint, and `npm test` for the inference-semantics tests.

## FastAPI interface

Set `VITE_API_BASE_URL` to the FastAPI origin. The frontend centralizes these routes in `src/services/api.ts`:

| Method and path | Expected response / request |
| --- | --- |
| `GET /api/health` | Backend and model availability, device, model/checkpoint identification, safe model error |
| `POST /api/analyze` | Original audio in multipart field `file`; real model scores, audio metadata, and inference provenance |

The API supports M4A, WAV, FLAC, MP3, OGG and AIFF up to 50 MB. It converts accepted recordings to mono 16 kHz PCM WAV server-side before passing them through the checkpoint-configured preprocessing path.

For file inference, the frontend only displays a classification when the response includes `detection.status: "AVAILABLE"`, a class agreeing with the two real class scores, finite scores summing to 1, the checkpoint class mapping, model/checkpoint/request provenance, model version, and processing time. Offline results display `Detection Status: UNAVAILABLE` and `AUTHENTICITY: NOT ANALYSED`; validation and filterbank compatibility never imply authenticity. Incomplete or inconsistent available responses are surfaced as errors.

Audio preflight is independent of inference and runs locally: it reports decoded sample rate, duration, SHA-256 of the selected file, and compatibility with the repository's 16 kHz LFCC preprocessing target. `Audio Validation: PASS` describes input processing only and never implies an authenticity decision.

## Implemented behavior

- Responsive six-page navigation with an animated sidebar, mobile drawer, saved theme/accessibility preferences, reduced-motion handling, and loading/empty/error states.
- Microphone visualization uses `getUserMedia`, `AudioContext`, and an `AnalyserNode` in the browser. It requests permission only after the user starts monitoring, selects a device when available, and stops all tracks when monitoring stops or the page unmounts. Audio processing preferences are saved locally and applied to the browser capture constraints. Audio is not uploaded by the monitoring page. Spoof scoring is unavailable until a real streaming/model endpoint exists.
- File selection accepts M4A, WAV, FLAC, MP3, OGG, and AIFF up to 50 MB. A local preview player, decoded waveform, SHA-256, duration, sampling-rate, and LFCC compatibility preflight are provided. The original file is uploaded only after the user requests inference while the backend and model are available. Classification and model scores are displayed only from a provenance-bearing available backend response.
- Dashboard and history show no fabricated records or metrics. The assistant sends to the configured API and reports unavailable errors instead of inventing responses. A completed analysis response is kept in app memory and can be passed as context to the assistant endpoint.

The frontend does not save microphone recordings or uploaded audio. Browser microphone access requires explicit permission and a secure context; localhost is permitted for development.
