package com.example.vectaar.transport

import android.content.Context
import android.media.AudioAttributes
import android.media.AudioDeviceInfo
import android.media.AudioFormat
import android.media.AudioManager
import android.media.AudioTrack
import android.os.Build
import android.util.Base64

/**
 * Plays the agent's speech: 16 kHz mono PCM16 chunks, streamed as they arrive.
 * WebRTC puts the phone in in-call audio mode once the mic track exists, which routes
 * playback to the earpiece; we pin the communication route to the loudspeaker instead.
 */
class TtsPlayer(context: Context) {
    private val rate = 16000
    private val audioManager = context.getSystemService(AudioManager::class.java)

    @Volatile var deafened = false   // drop the agent's speech entirely

    init { routeToSpeaker() }

    private fun routeToSpeaker() {
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

    private val track = AudioTrack.Builder()
        .setAudioAttributes(AudioAttributes.Builder().setUsage(AudioAttributes.USAGE_VOICE_COMMUNICATION)
            .setContentType(AudioAttributes.CONTENT_TYPE_SPEECH).build())
        .setAudioFormat(AudioFormat.Builder().setEncoding(AudioFormat.ENCODING_PCM_16BIT)
            .setSampleRate(rate).setChannelMask(AudioFormat.CHANNEL_OUT_MONO).build())
        .setBufferSizeInBytes(rate * 2 * 4)    // 4 s; chunks are 0.75 s
        .setTransferMode(AudioTrack.MODE_STREAM)
        .build()

    init { track.play() }

    fun play(chunk: TtsChunk) {
        if (deafened) return
        val pcm = Base64.decode(chunk.data, Base64.DEFAULT)
        if (pcm.isNotEmpty()) track.write(pcm, 0, pcm.size)   // blocks only if the buffer is full
    }

    fun release() {
        runCatching { track.stop() }; track.release()
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) audioManager.clearCommunicationDevice()
        audioManager.mode = AudioManager.MODE_NORMAL
    }
}
