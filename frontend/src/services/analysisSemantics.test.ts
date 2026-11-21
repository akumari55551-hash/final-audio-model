import { describe, expect, it } from 'vitest'
import { createAnalysisView, createOfflineAnalysis, normalizeDetectionResponse } from './analysisSemantics'
import type { AudioValidation } from '../types'

const validAudio: AudioValidation = { status: 'PASS', originalFormat: 'WAV', fileSizeBytes: 48000, sampleRateHz: 48000, channelCount: 2, durationSeconds: 3.25, sha256: 'a'.repeat(64), filterbankCompatibility: 'PASS' }
const response = (authenticity = 'SPOOF') => ({
  request_id: 'run-42', model_version: 'checkpoint-epoch-3', processing_time_ms: 31.2,
  bona_fide_score: authenticity === 'BONA_FIDE' ? 0.8 : 0.2,
  spoof_score: authenticity === 'BONA_FIDE' ? 0.2 : 0.8,
  detection: { status: 'AVAILABLE', authenticity, confidence: 0.8, bona_fide_score: authenticity === 'BONA_FIDE' ? 0.8 : 0.2, spoof_score: authenticity === 'BONA_FIDE' ? 0.2 : 0.8,
    provenance: { model_name: 'LFCCCNNDetector', model_version: 'checkpoint-epoch-3', checkpoint_sha256: 'b'.repeat(64), inference_id: 'run-42', class_mapping: { bona_fide: 0, spoof: 1 } } },
})

describe('audio validation and inference semantics', () => {
  it('keeps validation PASS separate from unavailable authenticity', () => {
    const result = createOfflineAnalysis(validAudio)
    expect(result.validation.status).toBe('PASS')
    expect(result.detection).toEqual({ status: 'UNAVAILABLE', authenticity: 'UNKNOWN' })
    expect(result.detection).not.toHaveProperty('confidence')
    expect(result.detection).not.toHaveProperty('spoofScore')
  })
  it('ignores standby predictions and legacy scores', () => {
    expect(createAnalysisView({ detection: { status: 'standby', authenticity: 'spoof', confidence: 0.98 }, spoof_score: 0.98 }, validAudio).detection).toEqual({ status: 'UNAVAILABLE', authenticity: 'UNKNOWN' })
  })
  it('renders backend scores and inference provenance only from a complete real response', () => {
    expect(normalizeDetectionResponse(response())).toMatchObject({ status: 'AVAILABLE', authenticity: 'SPOOF', spoofScore: 0.8, bonaFideScore: 0.2, processingTimeMs: 31.2, modelVersion: 'checkpoint-epoch-3', provenance: { modelName: 'LFCCCNNDetector', inferenceId: 'run-42' } })
  })
  it('accepts the backend genuine class when scores and mapping agree', () => {
    expect(normalizeDetectionResponse(response('BONA_FIDE')).authenticity).toBe('BONA_FIDE')
  })
  it('preserves per-window raw logits and softmax scores for diagnostics', () => {
    const raw = response()
    Object.assign(raw, { window_predictions: [{ window_index: 0, logits: [-1.2, 1.4], bona_fide_score: 0.069138, spoof_score: 0.930862, predicted_class: 'SPOOF' }] })
    expect(normalizeDetectionResponse(raw).windowPredictions).toEqual([{ windowIndex: 0, logits: [-1.2, 1.4], bonaFideScore: 0.069138, spoofScore: 0.930862, predictedClass: 'SPOOF' }])
  })
  it('rejects available but incomplete or malformed inference responses visibly', () => {
    expect(() => normalizeDetectionResponse({ detection: { status: 'AVAILABLE', authenticity: 'SPOOF' } })).toThrow(/incomplete or malformed/)
    const mismatch = response()
    mismatch.detection.authenticity = 'BONA_FIDE'
    expect(() => normalizeDetectionResponse(mismatch)).toThrow(/does not match/)
  })
})
