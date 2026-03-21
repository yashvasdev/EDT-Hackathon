import { Audio } from "expo-av"
import * as FileSystem from "expo-file-system"

const SAMPLE_RATE = 16000
const DURATION_SEC = 0.3
const FREQUENCY_HZ = 2500
const BEEP_COUNT = 3
const GAP_SEC = 0.15

let soundObject = null
let soundUri = null

function generateBeepWav() {
  const beepSamples = Math.floor(SAMPLE_RATE * DURATION_SEC)
  const gapSamples = Math.floor(SAMPLE_RATE * GAP_SEC)
  const totalSamples =
    BEEP_COUNT * beepSamples + (BEEP_COUNT - 1) * gapSamples

  const dataSize = totalSamples * 2 // 16-bit mono
  const fileSize = 44 + dataSize

  const buffer = new ArrayBuffer(fileSize)
  const view = new DataView(buffer)

  // WAV header
  writeString(view, 0, "RIFF")
  view.setUint32(4, fileSize - 8, true)
  writeString(view, 8, "WAVE")
  writeString(view, 12, "fmt ")
  view.setUint32(16, 16, true) // chunk size
  view.setUint16(20, 1, true) // PCM
  view.setUint16(22, 1, true) // mono
  view.setUint32(24, SAMPLE_RATE, true)
  view.setUint32(28, SAMPLE_RATE * 2, true) // byte rate
  view.setUint16(32, 2, true) // block align
  view.setUint16(34, 16, true) // bits per sample
  writeString(view, 36, "data")
  view.setUint32(40, dataSize, true)

  let offset = 44
  for (let b = 0; b < BEEP_COUNT; b++) {
    // Beep
    for (let i = 0; i < beepSamples; i++) {
      const t = i / SAMPLE_RATE
      const sample = Math.sin(2 * Math.PI * FREQUENCY_HZ * t) * 0.8
      view.setInt16(offset, sample * 32767, true)
      offset += 2
    }
    // Gap (silence) between beeps
    if (b < BEEP_COUNT - 1) {
      for (let i = 0; i < gapSamples; i++) {
        view.setInt16(offset, 0, true)
        offset += 2
      }
    }
  }

  // Convert to base64
  const bytes = new Uint8Array(buffer)
  let binary = ""
  for (let i = 0; i < bytes.length; i++) {
    binary += String.fromCharCode(bytes[i])
  }
  return btoa(binary)
}

function writeString(view, offset, str) {
  for (let i = 0; i < str.length; i++) {
    view.setUint8(offset + i, str.charCodeAt(i))
  }
}

export async function initAlertSound() {
  if (soundUri) return

  await Audio.setAudioModeAsync({
    allowsRecordingIOS: false,
    playsInSilentModeIOS: true,
    shouldDuckAndroid: false,
    playThroughEarpieceAndroid: false,
  })

  const base64 = generateBeepWav()
  soundUri = FileSystem.cacheDirectory + "alert_beep.wav"
  await FileSystem.writeAsStringAsync(soundUri, base64, {
    encoding: FileSystem.EncodingType.Base64,
  })
}

export async function playAlertSound() {
  try {
    if (!soundUri) await initAlertSound()

    // Unload previous if still loaded
    if (soundObject) {
      await soundObject.unloadAsync()
      soundObject = null
    }

    const { sound } = await Audio.Sound.createAsync(
      { uri: soundUri },
      { shouldPlay: true, volume: 1.0 },
    )
    soundObject = sound

    sound.setOnPlaybackStatusUpdate((status) => {
      if (status.didJustFinish) {
        sound.unloadAsync()
        soundObject = null
      }
    })
  } catch (err) {
    console.warn("Failed to play alert sound:", err)
  }
}
