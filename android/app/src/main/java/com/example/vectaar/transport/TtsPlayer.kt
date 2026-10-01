package com.example.vectaar.transport

import android.media.AudioAttributes
import android.media.AudioFormat
import android.media.AudioTrack
import android.util.Base64

/** Plays the agent's speech: 16 kHz mono PCM16 chunks, streamed as they arrive. */
class TtsPlayer {
    private val rate = 16000
    private val track = AudioTrack.Builder()
        .setAudioAttributes(AudioAttributes.Builder().setUsage(AudioAttributes.USAGE_ASSISTANT)
            .setContentType(AudioAttributes.CONTENT_TYPE_SPEECH).build())
        .setAudioFormat(AudioFormat.Builder().setEncoding(AudioFormat.ENCODING_PCM_16BIT)
            .setSampleRate(rate).setChannelMask(AudioFormat.CHANNEL_OUT_MONO).build())
        .setBufferSizeInBytes(rate * 2 * 4)    // 4 s; chunks are 0.75 s
        .setTransferMode(AudioTrack.MODE_STREAM)
        .build()

    init { track.play() }

    fun play(chunk: TtsChunk) {
        val pcm = Base64.decode(chunk.data, Base64.DEFAULT)
        if (pcm.isNotEmpty()) track.write(pcm, 0, pcm.size)   // blocks only if the buffer is full
    }

    fun release() { runCatching { track.stop() }; track.release() }
}
