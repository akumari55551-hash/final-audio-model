import { describe, expect, it, vi } from 'vitest'
import { readFile } from 'node:fs/promises'
import { getAudioFormat, MAX_AUDIO_FILE_BYTES, type AudioTranscoder } from './audioDecoder'
import { validateAudioFile } from './audioPreflight'

const fixtureData: Array<{ name: string; format: string; sampleRateHz: number; channels: number }> = [
  { name: 'sample.m4a', format: 'M4A', sampleRateHz: 44_100, channels: 2 },
  { name: 'sample.wav', format: 'WAV', sampleRateHz: 48_000, channels: 2 },
  { name: 'sample.flac', format: 'FLAC', sampleRateHz: 32_000, channels: 1 },
  { name: 'sample.mp3', format: 'MP3', sampleRateHz: 44_100, channels: 2 },
  { name: 'sample.ogg', format: 'OGG', sampleRateHz: 48_000, channels: 2 },
  { name: 'sample.aiff', format: 'AIFF', sampleRateHz: 44_100, channels: 2 },
]

async function fixtureBytes(name: string): Promise<Uint8Array> {
  return new Uint8Array(await readFile(new URL(`./fixtures/${name}`, import.meta.url)))
}

async function fileFromFixture(name: string): Promise<File> {
  return new File([await fixtureBytes(name)], name)
}

describe('six-format audio preflight', () => {
  it.each(fixtureData)('validates $format, hashes the original, and prepares mono 16 kHz PCM WAV', async (fixture) => {
    const original = await fileFromFixture(fixture.name)
    const wavBytes = await fixtureBytes('preprocessed-16khz.wav')
    const transcoder: AudioTranscoder = {
      transcode: vi.fn(async (_bytes, format) => {
        expect(format).toBe(fixture.format)
        return { sampleRateHz: fixture.sampleRateHz, channelCount: fixture.channels, durationSeconds: 0.25, wavBytes }
      }),
    }

    const { validation, preprocessedFile } = await validateAudioFile(original, transcoder)
    const sourceDigest = await crypto.subtle.digest('SHA-256', await original.arrayBuffer())
    const expectedHash = Array.from(new Uint8Array(sourceDigest), (byte) => byte.toString(16).padStart(2, '0')).join('')

    expect(validation.status).toBe('PASS')
    expect(validation.filterbankCompatibility).toBe('PASS')
    expect(validation.originalFormat).toBe(fixture.format)
    expect(validation.fileSizeBytes).toBe(original.size)
    expect(validation.sampleRateHz).toBe(fixture.sampleRateHz)
    expect(validation.channelCount).toBe(fixture.channels)
    expect(validation.durationSeconds).toBeCloseTo(0.25)
    expect(validation.sha256).toBe(expectedHash)
    expect(preprocessedFile).not.toBeNull()
    expect(preprocessedFile?.name).toBe('sample_16khz_mono.wav')
    expect(preprocessedFile?.type).toBe('audio/wav')
    const output = new DataView(await preprocessedFile!.arrayBuffer())
    expect(output.getUint32(0, false)).toBe(0x52494646)
    expect(output.getUint16(22, true)).toBe(1)
    expect(output.getUint32(24, true)).toBe(16_000)
    expect(output.getUint16(34, true)).toBe(16)
  })

  it('recognizes only the six requested extensions and their common MIME types', () => {
    expect(fixtureData.map(({ name }) => getAudioFormat({ name, type: '' }))).toEqual(fixtureData.map(({ format }) => format))
    expect(getAudioFormat({ name: 'recording.bin', type: 'audio/mpeg' })).toBe('MP3')
    expect(getAudioFormat({ name: 'recording.webm', type: 'audio/webm' })).toBeNull()
  })

  it('rejects empty, oversized, and unsupported files before decoding', async () => {
    const transcode = vi.fn()
    const transcoder = { transcode } as unknown as AudioTranscoder
    const empty = await validateAudioFile(new File([], 'empty.wav'), transcoder)
    const oversized = await validateAudioFile(new File([new Uint8Array(MAX_AUDIO_FILE_BYTES + 1)], 'large.wav'), transcoder)
    const unsupported = await validateAudioFile(new File(['data'], 'note.txt'), transcoder)

    expect(empty.validation.status).toBe('FAIL')
    expect(empty.validation.message).toMatch(/empty/i)
    expect(oversized.validation.status).toBe('FAIL')
    expect(oversized.validation.message).toMatch(/50 MB/i)
    expect(unsupported.validation.message).toMatch(/M4A.*WAV.*FLAC.*MP3.*OGG.*AIFF/)
    expect(transcode).not.toHaveBeenCalled()
  })

  it('reports corrupt and unsupported-codec decoding failures without a prediction file', async () => {
    const transcoder: AudioTranscoder = { transcode: vi.fn(async () => { throw new Error('This file uses an unsupported audio codec.') }) }
    const result = await validateAudioFile(new File(['not audio'], 'broken.flac'), transcoder)
    expect(result.validation.status).toBe('FAIL')
    expect(result.validation.filterbankCompatibility).toBe('FAIL')
    expect(result.validation.message).toMatch(/unsupported audio codec/i)
    expect(result.preprocessedFile).toBeNull()
  })
})
