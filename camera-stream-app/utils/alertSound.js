import { Audio } from "expo-av"

const ALERT_BEEP = require("../assets/alert_beep.wav")

let soundObject = null
let initialized = false

export async function initAlertSound() {
  if (initialized) return

  try {
    await Audio.setAudioModeAsync({
      allowsRecordingIOS: false,
      playsInSilentModeIOS: true,
      shouldDuckAndroid: false,
      playThroughEarpieceAndroid: false,
      staysActiveInBackground: true,
    })
    initialized = true
    console.log("[AlertSound] Audio mode configured")
  } catch (err) {
    console.error("[AlertSound] Failed to configure audio mode:", err)
  }
}

export async function playAlertSound() {
  try {
    if (!initialized) await initAlertSound()

    // Unload previous if still loaded
    if (soundObject) {
      await soundObject.unloadAsync()
      soundObject = null
    }

    const { sound } = await Audio.Sound.createAsync(ALERT_BEEP, {
      shouldPlay: true,
      volume: 1.0,
    })
    soundObject = sound
    console.log("[AlertSound] Playing alert beep")

    sound.setOnPlaybackStatusUpdate((status) => {
      if (status.didJustFinish) {
        sound.unloadAsync()
        soundObject = null
      }
    })
  } catch (err) {
    console.error("[AlertSound] Failed to play alert sound:", err)
  }
}
