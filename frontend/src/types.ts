export type BackendStatus = 'checking' | 'connected' | 'unavailable'
export interface BackendHealth {
  status: BackendStatus
  modelLoaded: boolean
  device?: string | null
  model?: { name: string; version: string; checkpoint_sha256: string } | null
  error?: string | null
}
export type Classification = 'bona_fide' | 'spoof' | 'unknown'
export type DetectionStatus = 'AVAILABLE' | 'UNAVAILABLE'
export type Authenticity = 'BONA_FIDE' | 'SPOOF' | 'UNKNOWN'

export interface AudioValidation {
  status: 'PASS' | 'FAIL'
  originalFormat: 'M4A' | 'WAV' | 'FLAC' | 'MP3' | 'OGG' | 'AIFF' | null
  fileSizeBytes: number
  sampleRateHz: number | null
  channelCount: number | null
  durationSeconds: number | null
  sha256: string | null
  filterbankCompatibility: 'PASS' | 'FAIL'
  message?: string
}

export interface InferenceProvenance {
  modelName?: string
  modelVersion?: string
  checkpointSha256?: string
  inferenceId?: string
  generatedAt?: string
  source?: string
  device?: string
  classMapping?: Record<string, number>
  scoreAggregation?: string
  evaluationStatus?: string
}

export interface WindowPrediction {
  windowIndex: number
  logits: [number, number]
  bonaFideScore: number
  spoofScore: number
  predictedClass: Authenticity
}

export interface AnalysisResponse {
  detection: {
    status: DetectionStatus
    authenticity: Authenticity
    confidence?: number
    provenance?: InferenceProvenance
    explanation?: string
    bonaFideScore?: number
    spoofScore?: number
    processingTimeMs?: number
    modelVersion?: string
    windowPredictions?: WindowPrediction[]
  }
  validation?: AudioValidation
}

export interface AnalysisViewResult {
  validation: AudioValidation
  detection: AnalysisResponse['detection']
}
export interface AnalysisRecord {
  id: string
  created_at: string
  source_name?: string
  classification: Classification
  risk_level?: 'low' | 'medium' | 'high' | 'unknown'
  status: 'completed' | 'processing' | 'failed'
  confidence?: number
  explanation?: string
}
export interface ChatResponse { answer: string }
export type PageKey = 'dashboard' | 'monitoring' | 'analysis' | 'history' | 'assistant' | 'settings'
