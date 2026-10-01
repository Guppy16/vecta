package com.example.vectaar.transport

import android.content.Context
import android.util.Log
import kotlinx.coroutines.CompletableDeferred
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.suspendCancellableCoroutine
import kotlinx.coroutines.withContext
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody
import org.json.JSONObject
import org.webrtc.Camera2Enumerator
import org.webrtc.CameraVideoCapturer
import org.webrtc.DataChannel
import org.webrtc.DefaultVideoDecoderFactory
import org.webrtc.DefaultVideoEncoderFactory
import org.webrtc.EglBase
import org.webrtc.IceCandidate
import org.webrtc.MediaConstraints
import org.webrtc.MediaStream
import org.webrtc.PeerConnection
import org.webrtc.PeerConnectionFactory
import org.webrtc.RTCStatsReport
import org.webrtc.RtpReceiver
import org.webrtc.SdpObserver
import org.webrtc.SessionDescription
import org.webrtc.SurfaceTextureHelper
import org.webrtc.VideoSink
import org.webrtc.VideoTrack
import java.nio.ByteBuffer
import java.nio.charset.StandardCharsets
import kotlin.coroutines.resume
import kotlin.coroutines.resumeWithException
import kotlin.time.Duration.Companion.seconds

private const val TAG = "RtcClient"

/** Outbound video numbers from WebRTC's own stats, sampled by the caller. */
data class RtcStats(
    val fps: Double? = null,
    val bytesSent: Long = 0,
    val framesEncoded: Long = 0,
    val width: Int? = null,
    val height: Int? = null,
    val codec: String? = null,
    val rttMs: Double? = null,          // ICE candidate-pair RTT
    val availableKbps: Double? = null,  // estimated outgoing bandwidth
)

/**
 * One WebRTC session to the server: the back camera as a video track, plus a
 * "control" data channel carrying [Message]s. Signalling is a single HTTP
 * offer/answer (no trickle ICE, no STUN/TURN — both ends are on the tailnet).
 * Single use: after [close], make a new instance to reconnect.
 */
class RtcClient(
    private val context: Context,
    private val serverUrl: String,          // e.g. http://100.77.155.9:8000
    private val eglBase: EglBase,           // shared with the preview renderer, outlives this client
    private val listener: Listener,
) {
    interface Listener {
        fun onMessage(msg: Message)
        fun onConnectionState(state: String)
    }

    private val http = OkHttpClient.Builder().readTimeout(30.seconds).build()
    private var factory: PeerConnectionFactory? = null
    private var capturer: CameraVideoCapturer? = null
    private var surfaceHelper: SurfaceTextureHelper? = null
    private var videoTrack: VideoTrack? = null
    private var pc: PeerConnection? = null
    private var channel: DataChannel? = null

    fun addSink(sink: VideoSink) { videoTrack?.addSink(sink) }

    /** Starts the camera and negotiates the connection. */
    suspend fun connect(sessionId: String? = null, width: Int = 1280, height: Int = 720, fps: Int = 30) {
        PeerConnectionFactory.initialize(
            PeerConnectionFactory.InitializationOptions.builder(context).createInitializationOptions()
        )
        val f = PeerConnectionFactory.builder()
            .setVideoEncoderFactory(DefaultVideoEncoderFactory(eglBase.eglBaseContext, true, true))
            .setVideoDecoderFactory(DefaultVideoDecoderFactory(eglBase.eglBaseContext))
            .createPeerConnectionFactory()
        factory = f

        val enumerator = Camera2Enumerator(context)
        val camera = enumerator.deviceNames.firstOrNull { enumerator.isBackFacing(it) }
            ?: enumerator.deviceNames.first()
        val cap = enumerator.createCapturer(camera, null)
        val helper = SurfaceTextureHelper.create("capture", eglBase.eglBaseContext)
        val source = f.createVideoSource(false)
        cap.initialize(helper, context, source.capturerObserver)
        cap.startCapture(width, height, fps)
        capturer = cap; surfaceHelper = helper
        val track = f.createVideoTrack("video0", source)
        videoTrack = track

        val config = PeerConnection.RTCConfiguration(emptyList()).apply {
            sdpSemantics = PeerConnection.SdpSemantics.UNIFIED_PLAN
        }
        val gathered = CompletableDeferred<Unit>()
        val peer = f.createPeerConnection(config, object : PeerObserver() {
            override fun onIceGatheringChange(state: PeerConnection.IceGatheringState) {
                if (state == PeerConnection.IceGatheringState.COMPLETE) gathered.complete(Unit)
            }
            override fun onConnectionChange(state: PeerConnection.PeerConnectionState) {
                listener.onConnectionState(state.name.lowercase())
            }
        }) ?: error("createPeerConnection returned null")
        pc = peer
        peer.addTrack(track, listOf("stream0"))
        // always-on microphone; the server segments and transcribes it
        val audioSource = f.createAudioSource(MediaConstraints())
        peer.addTrack(f.createAudioTrack("audio0", audioSource), listOf("stream0"))
        channel = peer.createDataChannel("control", DataChannel.Init()).apply {
            registerObserver(ChannelObserver())
        }

        val offer = peer.createOffer()
        peer.setLocalDescription(offer)
        gathered.await()                       // all candidates go in the SDP; no trickle
        val local = peer.localDescription ?: error("no local description")
        val answer = withContext(Dispatchers.IO) { signal(local, sessionId) }
        peer.setRemoteDescription(answer)
    }

    val channelOpen: Boolean get() = channel?.state() == DataChannel.State.OPEN

    fun send(msg: Message): Boolean {
        val ch = channel ?: return false
        if (ch.state() != DataChannel.State.OPEN) return false
        val bytes = Protocol.encode(msg).toByteArray(StandardCharsets.UTF_8)
        return ch.send(DataChannel.Buffer(ByteBuffer.wrap(bytes), false))
    }

    suspend fun stats(): RtcStats? {
        val peer = pc ?: return null
        val report = suspendCancellableCoroutine<RTCStatsReport> { cont -> peer.getStats { cont.resume(it) } }
        return parseStats(report)
    }

    fun close() {
        runCatching { capturer?.stopCapture() }
        channel?.close(); pc?.close()
        capturer?.dispose(); surfaceHelper?.dispose(); videoTrack?.dispose()
        factory?.dispose()
    }

    // --- signalling: POST the offer, get the answer ---

    private fun signal(offer: SessionDescription, sessionId: String?): SessionDescription {
        val body = JSONObject().put("sdp", offer.description).put("type", "offer")
            .put("session_id", sessionId ?: JSONObject.NULL).toString()   // resume our identity
        val req = Request.Builder().url("$serverUrl/rtc/offer")
            .post(body.toRequestBody("application/json".toMediaType())).build()
        http.newCall(req).execute().use { resp ->
            check(resp.isSuccessful) { "offer rejected: HTTP ${resp.code}" }
            val json = JSONObject(resp.body.string())
            return SessionDescription(SessionDescription.Type.ANSWER, json.getString("sdp"))
        }
    }

    // --- data channel ---

    private inner class ChannelObserver : DataChannel.Observer {
        override fun onBufferedAmountChange(previousAmount: Long) {}
        override fun onStateChange() { Log.d(TAG, "channel ${channel?.state()}") }
        override fun onMessage(buffer: DataChannel.Buffer) {
            val bytes = ByteArray(buffer.data.remaining()).also { buffer.data.get(it) }
            Protocol.decode(String(bytes, StandardCharsets.UTF_8))?.let(listener::onMessage)
        }
    }
}

/** Pull the numbers we show out of a standard WebRTC stats report. */
private fun parseStats(report: RTCStatsReport): RtcStats {
    val all = report.statsMap.values
    val out = all.firstOrNull { it.type == "outbound-rtp" && it.members["kind"] == "video" }
    val codecId = out?.members?.get("codecId") as? String
    val codec = codecId?.let { report.statsMap[it]?.members?.get("mimeType") as? String }
    val pair = all.firstOrNull { it.type == "candidate-pair" && it.members["state"] == "succeeded" }
    fun num(m: Map<String, Any>?, k: String): Double? = (m?.get(k) as? Number)?.toDouble()
    return RtcStats(
        fps = num(out?.members, "framesPerSecond"),
        bytesSent = num(out?.members, "bytesSent")?.toLong() ?: 0,
        framesEncoded = num(out?.members, "framesEncoded")?.toLong() ?: 0,
        width = num(out?.members, "frameWidth")?.toInt(),
        height = num(out?.members, "frameHeight")?.toInt(),
        codec = codec?.removePrefix("video/"),
        rttMs = num(pair?.members, "currentRoundTripTime")?.let { it * 1000 },
        availableKbps = num(pair?.members, "availableOutgoingBitrate")?.let { it / 1000 },
    )
}

/** Suspending wrappers over the callback-style WebRTC API. */
private suspend fun PeerConnection.createOffer(): SessionDescription =
    suspendCancellableCoroutine { cont ->
        createOffer(object : SdpAdapter() {
            override fun onCreateSuccess(sdp: SessionDescription) { cont.resume(sdp) }
            override fun onCreateFailure(error: String) { cont.resumeWithException(IllegalStateException(error)) }
        }, MediaConstraints())
    }

private suspend fun PeerConnection.setLocalDescription(sdp: SessionDescription): Unit =
    suspendCancellableCoroutine { cont ->
        setLocalDescription(object : SdpAdapter() {
            override fun onSetSuccess() { cont.resume(Unit) }
            override fun onSetFailure(error: String) { cont.resumeWithException(IllegalStateException(error)) }
        }, sdp)
    }

private suspend fun PeerConnection.setRemoteDescription(sdp: SessionDescription): Unit =
    suspendCancellableCoroutine { cont ->
        setRemoteDescription(object : SdpAdapter() {
            override fun onSetSuccess() { cont.resume(Unit) }
            override fun onSetFailure(error: String) { cont.resumeWithException(IllegalStateException(error)) }
        }, sdp)
    }

private open class SdpAdapter : SdpObserver {
    override fun onCreateSuccess(sdp: SessionDescription) {}
    override fun onSetSuccess() {}
    override fun onCreateFailure(error: String) {}
    override fun onSetFailure(error: String) {}
}

/** No-op implementation so callers override only what they need. */
private open class PeerObserver : PeerConnection.Observer {
    override fun onSignalingChange(state: PeerConnection.SignalingState) {}
    override fun onIceConnectionChange(state: PeerConnection.IceConnectionState) {}
    override fun onIceConnectionReceivingChange(receiving: Boolean) {}
    override fun onIceGatheringChange(state: PeerConnection.IceGatheringState) {}
    override fun onIceCandidate(candidate: IceCandidate) {}
    override fun onIceCandidatesRemoved(candidates: Array<out IceCandidate>) {}
    override fun onAddStream(stream: MediaStream) {}
    override fun onRemoveStream(stream: MediaStream) {}
    override fun onDataChannel(channel: DataChannel) {}
    override fun onRenegotiationNeeded() {}
    override fun onAddTrack(receiver: RtpReceiver, streams: Array<out MediaStream>) {}
}
