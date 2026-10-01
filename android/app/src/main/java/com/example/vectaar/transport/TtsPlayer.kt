package com.example.vectaar.transport

import android.content.Context
import android.media.AudioAttributes
import android.media.AudioDeviceInfo
import android.media.AudioFormat
import android.media.AudioManager
import android.media.AudioTrack
import android.os.Build
import android.util.Base64
import android.util.Log

/**
 * Plays the agent's speech: 16 kHz mono PCM16 chunks, streamed as they arrive.
 *
 * WebRTC puts the phone in in-call audio mode once the mic track exists, which routes
 * playback to the earpiece; we pin the communication route to the loudspeaker. Route
 * changes (reconnects, mic restarts) can also kill an AudioTrack mid-life, so a fresh
 * track is used per utterance and every write is checked.
 */
class TtsPlayer(context: Context) {
    private val rate = 16000
    private val audioManager = context.getSystemService(AudioManager::class.java)
    @Volatile var deafened = false   // drop the agent's speech entirely
    private var track: AudioTrack? = null
    private var currentId: String? = null

    fun routeToSpeaker() {
        audioManager.mode = AudioManager.MODE_IN_COMMUNICATION
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
            audioManager.availableCommunicationDevices
                .firstOrNull { it.type == AudioDeviceInfo.TYPE_BUILTIN_SPEAKER }
                ?.let { audioManager.setCommunicationDevice(it) }
        } else {
            @Suppress("DEPRECATION")
            audioManager.isSpeakerphoneOn = true
        }
        // our track plays on the voice-call stream, whose speaker volume is often left low
        val stream = AudioManager.STREAM_VOICE_CALL
        audioManager.setStreamVolume(stream, audioManager.getStreamMaxVolume(stream), 0)
    }

    @Synchronized
    fun play(chunk: TtsChunk) {
        if (deafened) return
        if (chunk.id != currentId || track == null) {
            track?.release()
            track = newTrack().also { it.play() }
            currentId = chunk.id
        }
        val pcm = Base64.decode(chunk.data, Base64.DEFAULT)
        if (pcm.isEmpty()) return
        var written = track!!.write(pcm, 0, pcm.size)
        if (written < 0) {                      // ERROR_DEAD_OBJECT etc.: the route changed under us
            Log.w("TtsPlayer", "write failed ($written); recreating track")
            track?.release()
            routeToSpeaker()
            track = newTrack().also { it.play() }
            written = track!!.write(pcm, 0, pcm.size)
            if (written < 0) Log.e("TtsPlayer", "write failed again ($written); dropping chunk")
        }
    }

    private fun newTrack() = AudioTrack.Builder()
        .setAudioAttributes(AudioAttributes.Builder().setUsage(AudioAttributes.USAGE_VOICE_COMMUNICATION)
            .setContentType(AudioAttributes.CONTENT_TYPE_SPEECH).build())
        .setAudioFormat(AudioFormat.Builder().setEncoding(AudioFormat.ENCODING_PCM_16BIT)
            .setSampleRate(rate).setChannelMask(AudioFormat.CHANNEL_OUT_MONO).build())
        .setBufferSizeInBytes(rate * 2 * 8)    // 8 s; a whole short reply fits without blocking
        .setTransferMode(AudioTrack.MODE_STREAM)
        .build()

    fun release() {
        track?.release(); track = null
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) audioManager.clearCommunicationDevice()
        audioManager.mode = AudioManager.MODE_NORMAL
    }
}
