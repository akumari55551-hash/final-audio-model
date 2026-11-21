import { FFmpeg } from '@ffmpeg/ffmpeg'
import coreURL from '@ffmpeg/core?url'
import wasmURL from '@ffmpeg/core/wasm?url'

export const SUPPORTED_AUDIO_FORMATS = ['M4A', 'WAV', 'FLAC', 'MP3', 'OGG', 'AIFF'] as const
export type SupportedAudioFormat = typeof SUPPORTED_AUDIO_FORMATS[number]
export const MAX_AUDIO_FILE_BYTES = 50 * 1024 * 1024
export const TARGET_SAMPLE_RATE_HZ = 16_000

const extensionFormats: Record<string, SupportedAudioFormat> = {
  '.m4a': 'M4A', '.wav': 'WAV', '.flac': 'FLAC', '.mp3': 'MP3', '.ogg': 'OGG', '.aiff': 'AIFF', '.aif': 'AIFF',
}
const mimeFormats: Record<string, SupportedAudioFormat> = {
  'audio/mp4': 'M4A', 'audio/x-m4a': 'M4A', 'audio/wav': 'WAV', 'audio/x-wav': 'WAV', 'audio/vnd.wave': 'WAV',
  'audio/flac': 'FLAC', 'audio/x-flac': 'FLAC', 'audio/mpeg': 'MP3', 'audio/mp3': 'MP3', 'audio/ogg': 'OGG',
  'audio/aiff': 'AIFF', 'audio/x-aiff': 'AIFF',
}

export function getAudioFormat(file: Pick<File, 'name' | 'type'>): SupportedAudioFormat | null {
  const extension = file.name.match(/\.[^.]+$/)?.[0]?.toLowerCase()
  return (extension && extensionFormats[extension]) || mimeFormats[file.type.toLowerCase()] || null
}

type SourceAudioMetadata = { sampleRateHz: number; channelCount: number; durationSeconds: number }
export type TranscodedAudio = SourceAudioMetadata & { wavBytes: Uint8Array }

export interface AudioTranscoder {
  transcode(fileBytes: ArrayBuffer, format: SupportedAudioFormat): Promise<TranscodedAudio>
}

let ffmpegPromise: Promise<FFmpeg> | null = null
let conversionQueue: Promise<void> = Promise.resolve()

async function getFFmpeg(): Promise<FFmpeg> {
  if (!ffmpegPromise) {
    ffmpegPromise = (async () => {
      const ffmpeg = new FFmpeg()
      await ffmpeg.load({ coreURL, wasmURL })
      return ffmpeg
    })().catch((error: unknown) => {
      ffmpegPromise = null
      throw error
    })
  }
  return ffmpegPromise
}

function parseWav(bytes: Uint8Array): { sampleRateHz: number; channelCount: number; bitsPerSample: number; pcm: boolean; dataBytes: number } {
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength)
  if (bytes.length < 44 || view.getUint32(0, false) !== 0x52494646 || view.getUint32(8, false) !== 0x57415645) {
    throw new Error('The audio decoder did not produce a valid WAV file.')
  }
  let offset = 12
  let sampleRateHz = 0
  let channelCount = 0
  let bitsPerSample = 0
  let pcm = false
  let dataBytes = 0
  while (offset + 8 <= bytes.length) {
    const chunkSize = view.getUint32(offset + 4, true)
    if (view.getUint32(offset, false) === 0x666d7420 && chunkSize >= 16) {
      pcm = view.getUint16(offset + 8, true) === 1
      channelCount = view.getUint16(offset + 10, true)
      sampleRateHz = view.getUint32(offset + 12, true)
      bitsPerSample = view.getUint16(offset + 22, true)
    } else if (view.getUint32(offset, false) === 0x64617461) dataBytes = chunkSize
    offset = offset + 8 + chunkSize + (chunkSize % 2)
  }
  if (!pcm || !channelCount || !sampleRateHz || !bitsPerSample || !dataBytes) throw new Error('The converted audio is not valid PCM WAV data.')
  return { sampleRateHz, channelCount, bitsPerSample, pcm, dataBytes }
}

class FFmpegAudioTranscoder implements AudioTranscoder {
  async transcode(fileBytes: ArrayBuffer, format: SupportedAudioFormat): Promise<TranscodedAudio> {
    const ffmpeg = await getFFmpeg()
    const inputPath = `original.${format.toLowerCase()}`
    const outputPath = 'preprocessed.wav'
    const metadataPath = 'source-metadata.json'
    const logs: string[] = []
    const onLog = ({ message }: { message: string }) => logs.push(message)
    ffmpeg.on('log', onLog)
    try {
      await ffmpeg.writeFile(inputPath, new Uint8Array(fileBytes))
      const probeCode = await ffmpeg.ffprobe([
        '-v', 'error', '-select_streams', 'a:0', '-show_entries', 'stream=sample_rate,channels,duration:format=duration',
        '-of', 'json', inputPath, '-o', metadataPath,
      ])
      if (probeCode !== 0) throw new Error('This file is corrupted or contains an unsupported audio codec.')
      const metadataRaw = await ffmpeg.readFile(metadataPath)
      if (typeof metadataRaw === 'string') throw new Error('Audio metadata could not be read.')
      const metadata = JSON.parse(new TextDecoder().decode(metadataRaw)) as {
        streams?: Array<{ sample_rate?: string; channels?: number; duration?: string }>
        format?: { duration?: string }
      }
      const stream = metadata.streams?.[0]
      const sampleRateHz = Number(stream?.sample_rate)
      const channelCount = Number(stream?.channels)
      const durationSeconds = Number(stream?.duration ?? metadata.format?.duration)
      if (!Number.isFinite(sampleRateHz) || sampleRateHz <= 0 || !Number.isInteger(channelCount) || channelCount <= 0) {
        throw new Error('The audio codec does not expose valid sample rate and channel metadata.')
      }

      const convertCode = await ffmpeg.exec([
        '-hide_banner', '-i', inputPath, '-map', '0:a:0', '-vn', '-ac', '1', '-ar', String(TARGET_SAMPLE_RATE_HZ),
        '-c:a', 'pcm_s16le', '-f', 'wav', outputPath,
      ])
      if (convertCode !== 0) throw new Error('Audio conversion failed. Check that the file uses a supported codec.')
      const converted = await ffmpeg.readFile(outputPath)
      if (typeof converted === 'string') throw new Error('The audio conversion returned invalid data.')
      const wavBytes = new Uint8Array(converted)
      const wav = parseWav(wavBytes)
      if (wav.sampleRateHz !== TARGET_SAMPLE_RATE_HZ || wav.channelCount !== 1 || wav.bitsPerSample !== 16 || !wav.pcm) {
        throw new Error('The converted file does not match the mono 16 kHz PCM preprocessing profile.')
      }
      const convertedDuration = wav.dataBytes / (TARGET_SAMPLE_RATE_HZ * 2)
      return {
        sampleRateHz,
        channelCount,
        durationSeconds: Number.isFinite(durationSeconds) && durationSeconds > 0 ? durationSeconds : convertedDuration,
        wavBytes,
      }
    } catch (error) {
      const details = logs.join('\n')
      if (error instanceof Error && /unsupported codec|corrupt|metadata|conversion|valid WAV|valid PCM/i.test(error.message)) throw error
      if (/unknown decoder|decoder .* not found|unsupported codec/i.test(details)) throw new Error('This file uses an unsupported audio codec.')
      throw new Error('The audio file is corrupted, unsupported, or could not be decoded.')
    } finally {
      ffmpeg.off('log', onLog)
      await Promise.all([inputPath, outputPath, metadataPath].map((path) => ffmpeg.deleteFile(path).catch(() => undefined)))
    }
  }
}

export const audioTranscoder: AudioTranscoder = {
  transcode(fileBytes, format) {
    const job = conversionQueue.then(() => new FFmpegAudioTranscoder().transcode(fileBytes, format))
    conversionQueue = job.then(() => undefined, () => undefined)
    return job
  },
}
