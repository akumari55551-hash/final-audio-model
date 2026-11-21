import type { AudioValidation } from '../types'
import { audioTranscoder, getAudioFormat, MAX_AUDIO_FILE_BYTES, TARGET_SAMPLE_RATE_HZ, type AudioTranscoder } from './audioDecoder'

export const LFCC_TARGET_SAMPLE_RATE_HZ = TARGET_SAMPLE_RATE_HZ

export type PreparedAudio = { validation: AudioValidation; preprocessedFile: File | null }

function bytesToHex(bytes: ArrayBuffer): string {
  return Array.from(new Uint8Array(bytes), (byte) => byte.toString(16).padStart(2, '0')).join('')
}

function wavFileName(fileName: string): string {
  return `${fileName.replace(/\.[^.]+$/, '') || 'audio'}_16khz_mono.wav`
}

/** Hashes the untouched upload, then decodes and converts it locally for model preprocessing. */
export async function validateAudioFile(file: File, transcoder: AudioTranscoder = audioTranscoder): Promise<PreparedAudio> {
  const originalFormat = getAudioFormat(file)
  const base: AudioValidation = {
    status: 'FAIL',
    originalFormat,
    fileSizeBytes: file.size,
    sampleRateHz: null,
    channelCount: null,
    durationSeconds: null,
    sha256: null,
    filterbankCompatibility: 'FAIL',
  }
  if (!originalFormat) return { validation: { ...base, message: 'Choose an M4A, WAV, FLAC, MP3, OGG, or AIFF audio file.' }, preprocessedFile: null }
  if (file.size === 0) return { validation: { ...base, message: 'The selected audio file is empty.' }, preprocessedFile: null }
  if (file.size > MAX_AUDIO_FILE_BYTES) return { validation: { ...base, message: 'This file exceeds the 50 MB limit.' }, preprocessedFile: null }

  let originalBytes: ArrayBuffer
  let sha256: string
  try {
    originalBytes = await file.arrayBuffer()
    if (!globalThis.crypto?.subtle) throw new Error('SHA-256 is unavailable in this browser context.')
    sha256 = bytesToHex(await globalThis.crypto.subtle.digest('SHA-256', originalBytes))
  } catch (error) {
    return { validation: { ...base, message: error instanceof Error ? error.message : 'The original file could not be read.' }, preprocessedFile: null }
  }

  try {
    const decoded = await transcoder.transcode(originalBytes, originalFormat)
    const durationSeconds = decoded.durationSeconds
    if (!Number.isFinite(durationSeconds) || durationSeconds <= 0) throw new Error('The audio file has an invalid duration.')
    const wavBuffer = decoded.wavBytes.slice().buffer as ArrayBuffer
    const preprocessedFile = new File([wavBuffer], wavFileName(file.name), { type: 'audio/wav' })
    return {
      validation: {
        ...base,
        status: 'PASS',
        sampleRateHz: decoded.sampleRateHz,
        channelCount: decoded.channelCount,
        durationSeconds,
        sha256,
        filterbankCompatibility: 'PASS',
      },
      preprocessedFile,
    }
  } catch (error) {
    return {
      validation: { ...base, sha256, message: error instanceof Error ? error.message : 'The audio file could not be decoded or converted.' },
      preprocessedFile: null,
    }
  }
}

export { MAX_AUDIO_FILE_BYTES }
