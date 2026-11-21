import type { AnalysisResponse, AnalysisViewResult, AudioValidation, Authenticity, InferenceProvenance, WindowPrediction } from '../types'

export const OFFLINE_DETECTION: AnalysisResponse['detection'] = Object.freeze({ status: 'UNAVAILABLE', authenticity: 'UNKNOWN' })

const recordOf = (value: unknown): Record<string, unknown> | null => value !== null && typeof value === 'object' && !Array.isArray(value) ? value as Record<string, unknown> : null
const text = (value: unknown): string | undefined => typeof value === 'string' && value.trim() ? value.trim() : undefined
const finiteScore = (value: unknown): value is number => typeof value === 'number' && Number.isFinite(value) && value >= 0 && value <= 1

function normalizeProvenance(value: unknown): InferenceProvenance | undefined {
  const source = recordOf(value)
  if (!source) return undefined
  const classMapping = recordOf(source.classMapping ?? source.class_mapping)
  return {
    ...(text(source.modelName ?? source.model_name) ? { modelName: text(source.modelName ?? source.model_name) } : {}),
    ...(text(source.modelVersion ?? source.model_version) ? { modelVersion: text(source.modelVersion ?? source.model_version) } : {}),
    ...(text(source.checkpointSha256 ?? source.checkpoint_sha256) ? { checkpointSha256: text(source.checkpointSha256 ?? source.checkpoint_sha256) } : {}),
    ...(text(source.inferenceId ?? source.inference_id) ? { inferenceId: text(source.inferenceId ?? source.inference_id) } : {}),
    ...(text(source.generatedAt ?? source.generated_at) ? { generatedAt: text(source.generatedAt ?? source.generated_at) } : {}),
    ...(text(source.source) ? { source: text(source.source) } : {}),
    ...(text(source.device) ? { device: text(source.device) } : {}),
    ...(classMapping ? { classMapping: classMapping as Record<string, number> } : {}),
    ...(text(source.scoreAggregation ?? source.score_aggregation) ? { scoreAggregation: text(source.scoreAggregation ?? source.score_aggregation) } : {}),
    ...(text(source.evaluationStatus ?? source.evaluation_status) ? { evaluationStatus: text(source.evaluationStatus ?? source.evaluation_status) } : {}),
  }
}

function normalizeAuthenticity(value: unknown): Authenticity {
  if (typeof value !== 'string') return 'UNKNOWN'
  const normalized = value.trim().toUpperCase().replace(/-/g, '_')
  if (normalized === 'BONA_FIDE' || normalized === 'GENUINE') return 'BONA_FIDE'
  if (normalized === 'SPOOF' || normalized === 'AI_GENERATED') return 'SPOOF'
  return 'UNKNOWN'
}

/** Only a complete, provenance-bearing real backend inference is classified. */
export function normalizeDetectionResponse(raw: unknown): AnalysisResponse['detection'] {
  const root = recordOf(raw)
  const detection = recordOf(root?.detection)
  if (!detection || text(detection.status)?.toUpperCase() !== 'AVAILABLE') return OFFLINE_DETECTION

  const provenance = normalizeProvenance(detection.provenance ?? root?.provenance)
  const mapping = provenance?.classMapping
  const bonaFideScore = detection.bona_fide_score ?? root?.bona_fide_score
  const spoofScore = detection.spoof_score ?? root?.spoof_score
  const authenticity = normalizeAuthenticity(detection.authenticity)
  const processingTimeMs = root?.processing_time_ms
  const modelVersion = text(root?.model_version) ?? provenance?.modelVersion
  const confidence = detection.confidence
  const windowPredictionsRaw = root?.window_predictions

  if (!provenance || !provenance.modelName || !provenance.checkpointSha256 || !provenance.inferenceId ||
      mapping?.bona_fide !== 0 || mapping?.spoof !== 1 || !finiteScore(bonaFideScore) || !finiteScore(spoofScore) ||
      Math.abs(bonaFideScore + spoofScore - 1) > 1e-4 || typeof processingTimeMs !== 'number' || !Number.isFinite(processingTimeMs) || processingTimeMs < 0 || !modelVersion) {
    throw new Error('The inference service returned an incomplete or malformed model response. No authenticity result is available.')
  }

  const expected = bonaFideScore === spoofScore ? 'UNKNOWN' : bonaFideScore > spoofScore ? 'BONA_FIDE' : 'SPOOF'
  if (authenticity !== expected) throw new Error('The inference response classification does not match its reported class scores.')

  const windowPredictions: WindowPrediction[] | undefined = Array.isArray(windowPredictionsRaw) ? windowPredictionsRaw.map((value, index) => {
    const item = recordOf(value)
    const logits = item?.logits
    const windowIndex = item?.window_index
    const windowBonaFide = item?.bona_fide_score
    const windowSpoof = item?.spoof_score
    const predictedClass = normalizeAuthenticity(item?.predicted_class)
    if (!item || windowIndex !== index || !Array.isArray(logits) || logits.length !== 2 || !logits.every((logit) => typeof logit === 'number' && Number.isFinite(logit)) ||
        !finiteScore(windowBonaFide) || !finiteScore(windowSpoof) || Math.abs(windowBonaFide + windowSpoof - 1) > 1e-4 ||
        predictedClass !== (logits[0] === logits[1] ? 'UNKNOWN' : logits[0] > logits[1] ? 'BONA_FIDE' : 'SPOOF')) {
      throw new Error(`Inference response contains a malformed window diagnostic at index ${index}.`)
    }
    return { windowIndex, logits: [logits[0], logits[1]], bonaFideScore: windowBonaFide, spoofScore: windowSpoof, predictedClass }
  }) : undefined

  return {
    status: 'AVAILABLE', authenticity, bonaFideScore, spoofScore, processingTimeMs, modelVersion,
    ...(finiteScore(confidence) ? { confidence } : {}), provenance,
    ...(windowPredictions ? { windowPredictions } : {}),
    ...(text(detection.explanation) ? { explanation: text(detection.explanation) } : {}),
  }
}

export function createOfflineAnalysis(validation: AudioValidation): AnalysisViewResult { return { validation, detection: OFFLINE_DETECTION } }
export function createAnalysisView(raw: unknown, validation: AudioValidation): AnalysisViewResult { return { validation, detection: normalizeDetectionResponse(raw) } }
