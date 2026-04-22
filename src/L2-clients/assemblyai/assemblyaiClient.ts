import { AssemblyAI } from 'assemblyai'
import { fileExistsSync, getFileStatsSync } from '../../L1-infra/fileSystem/fileSystem.js'
import { getConfig } from '../../L1-infra/config/environment'
import logger from '../../L1-infra/logger/configLogger'
import { getWhisperPrompt } from '../../L1-infra/config/brand'
import { Transcript, Segment, Word } from '../../L0-pure/types/index'

const MAX_FILE_SIZE_MB = 100
const WARN_FILE_SIZE_MB = 80
const MAX_RETRIES = 3
const RETRY_DELAY_MS = 5000

export async function transcribeAudio(audioPath: string): Promise<Transcript> {
  logger.info(`Starting AssemblyAI transcription: ${audioPath}`)

  if (!fileExistsSync(audioPath)) {
    throw new Error(`Audio file not found: ${audioPath}`)
  }

  const stats = getFileStatsSync(audioPath)
  const fileSizeMB = stats.size / (1024 * 1024)

  if (fileSizeMB > MAX_FILE_SIZE_MB) {
    throw new Error(
      `Audio file exceeds AssemblyAI's ${MAX_FILE_SIZE_MB}MB limit (${fileSizeMB.toFixed(1)}MB). ` +
      'The file should be split into smaller chunks before transcription.'
    )
  }
  if (fileSizeMB > WARN_FILE_SIZE_MB) {
    logger.warn(`Audio file is ${fileSizeMB.toFixed(1)}MB — approaching ${MAX_FILE_SIZE_MB}MB limit`)
  }

  const config = getConfig()
  const apiKeys = [config.ASSEMBLYAI_API_KEY, ...config.ASSEMBLYAI_API_KEYS].filter(Boolean)

  if (!apiKeys[0]) {
    throw new Error('ASSEMBLYAI_API_KEY is required. Get one at assemblyai.com')
  }

  let lastError: Error | undefined

  for (let keyIndex = 0; keyIndex < apiKeys.length; keyIndex++) {
    const apiKey = apiKeys[keyIndex]

    for (let attempt = 1; attempt <= MAX_RETRIES; attempt++) {
      try {
        const client = new AssemblyAI({ apiKey })

        const prompt = getWhisperPrompt()

        logger.info(`Transcribing with AssemblyAI (key ${keyIndex + 1}/${apiKeys.length}, attempt ${attempt}/${MAX_RETRIES})`)

        const transcript = await client.transcripts.transcribe({
          audio: audioPath,
          speech_models: ['universal-3'],
          word_boost: prompt ? [prompt] : undefined,
        })

        if (transcript.status === 'error') {
          const errObj = transcript.error as { message?: string } | string | undefined
          const errMsg = typeof errObj === 'string' ? errObj : errObj?.message
          throw new Error(errMsg || 'AssemblyAI transcription failed')
        }

        const words: Word[] = (transcript.words || []).map((w) => ({
          word: w.text,
          start: (w.start || 0) / 1000,
          end: (w.end || 0) / 1000,
        }))

        const segments: Segment[] = (transcript.utterances || []).map((u, id) => ({
          id,
          text: u.text || '',
          start: (u.start || 0) / 1000,
          end: (u.end || 0) / 1000,
          words: words.filter((w) => w.start >= (u.start || 0) / 1000 && w.end <= (u.end || 0) / 1000),
        }))

        if (segments.length === 0 && words.length > 0) {
          const minStart = Math.min(...words.map(w => w.start))
          const maxEnd = Math.max(...words.map(w => w.end))
          segments.push({
            id: 0,
            text: transcript.text || '',
            start: minStart,
            end: maxEnd,
            words,
          })
        }

        logger.info(
          `AssemblyAI transcription complete — ${segments.length} segments, ` +
          `${words.length} words, language=${transcript.language_code}`
        )

        return {
          text: transcript.text || '',
          segments,
          words,
          language: transcript.language_code || 'unknown',
          duration: (transcript.audio_duration || 0) / 1000,
        }
      } catch (retryError: unknown) {
        const status = typeof retryError === 'object' && retryError !== null && 'status' in retryError
          ? (retryError as { status?: number }).status
          : undefined

        if (status === 401 || status === 403) {
          logger.warn(`AssemblyAI key ${keyIndex + 1} failed with auth error, trying next key`)
          break
        }
        if (attempt === MAX_RETRIES) {
          lastError = retryError instanceof Error ? retryError : new Error(String(retryError))
          continue
        }

        const msg = retryError instanceof Error ? retryError.message : String(retryError)
        logger.warn(`AssemblyAI attempt ${attempt}/${MAX_RETRIES} failed: ${msg} — retrying in ${RETRY_DELAY_MS / 1000}s`)
        await new Promise(resolve => setTimeout(resolve, RETRY_DELAY_MS))
      }
    }
  }

  const message = lastError?.message || 'All AssemblyAI keys exhausted'
  logger.error(`AssemblyAI transcription failed: ${message}`)
  throw new Error(`AssemblyAI transcription failed: ${message}`)
}
