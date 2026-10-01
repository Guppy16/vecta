package com.example.vectaar

import android.app.Application
import android.util.Log
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.example.vectaar.transport.AgentMessage
import com.example.vectaar.transport.Capture
import com.example.vectaar.transport.CaptureAck
import com.example.vectaar.transport.CaptureRequest
import com.example.vectaar.transport.InputText
import com.example.vectaar.transport.MarkerSpec
import com.example.vectaar.transport.OverlaySet
import com.example.vectaar.transport.Transcript
import com.example.vectaar.transport.TtsChunk
import com.example.vectaar.transport.TtsPlayer
import com.example.vectaar.transport.Message
import com.example.vectaar.transport.PageRender
import com.example.vectaar.transport.Ping
import com.example.vectaar.transport.Pong
import com.example.vectaar.transport.RtcClient
import com.example.vectaar.transport.RtcStats
import com.example.vectaar.transport.SessionStart
import com.example.vectaar.transport.SessionState
import com.example.vectaar.transport.Status
import com.example.vectaar.transport.TaskSet
import com.example.vectaar.transport.UiEvent
import kotlinx.coroutines.CompletableDeferred
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import kotlinx.serialization.json.JsonElement
import org.webrtc.EglBase
import org.webrtc.VideoFrame
import org.webrtc.VideoSink

/** One row of the conversation. role: user | agent | system. */
data class ChatItem(
    val id: Long,
    val role: String,
    val text: String,
    val status: String = "",          // agent rows: searching | found | info | answer
    val imageUrl: String? = null,
    val page: PageRender? = null,
    val latencyMs: Int? = null,
    val ts: Long = System.currentTimeMillis(),
)

data class CaptureItem(val id: String, val url: String, val frames: Int, val ts: Long)

/** Everything the UI shows, in one immutable snapshot. */
data class UiState(
    val connection: String = "connecting",
    val sessionId: String? = null,
    val task: String? = null,
    val phase: String = "idle",
    val statusText: String = "",
    val rttMs: Long? = null,           // control-channel ping RTT
    val hint: String? = null,          // capture.request hint
    val messages: List<ChatItem> = emptyList(),
    val captures: List<CaptureItem> = emptyList(),
    val stats: RtcStats? = null,
    val kbps: Double? = null,          // bytesSent delta over the sampling interval
    val expandedPage: PageRender? = null,
    val micMuted: Boolean = false,
    val deafened: Boolean = false,
    val markers: List<MarkerSpec> = emptyList(),   // live overlay, image-normalised coords
    val frameW: Int = 0,                          // image size the markers refer to
    val frameH: Int = 0,
)

/**
 * Owns the connection (with reconnect) and turns server messages into [UiState].
 * The preview renderer attaches to [previewSink], which survives reconnects.
 */
class SessionViewModel(app: Application) : AndroidViewModel(app) {
    val serverUrl = "http://${BuildConfig.SERVER_IP}"
    val eglBase: EglBase = EglBase.create()
    val previewSink = ProxySink()

    private val prefs = app.getSharedPreferences("vecta", android.content.Context.MODE_PRIVATE)
    private val _state = MutableStateFlow(UiState(sessionId = prefs.getString("session_id", null)))
    val state: StateFlow<UiState> = _state
    private var rtc: RtcClient? = null
    private val tts = TtsPlayer(app)
    private var nextId = 1L

    init { viewModelScope.launch { connectLoop() } }

    // --- user actions ---

    fun submit(text: String) {
        if (text.isBlank()) return
        if (_state.value.task == null) setTask(text) else ask(text)
    }

    fun setTask(text: String) {
        rtc?.send(TaskSet(text))
        _state.update { it.copy(task = text) }
        append(ChatItem(nextId++, "user", text, status = "task"))
    }

    fun ask(text: String) {
        rtc?.send(InputText(text))
        append(ChatItem(nextId++, "user", text))
    }

    fun toggleMic() {
        val muted = !_state.value.micMuted
        rtc?.setMicEnabled(!muted)
        _state.update { it.copy(micMuted = muted) }
    }

    fun toggleDeafen() {
        val deaf = !_state.value.deafened
        tts.deafened = deaf
        rtc?.setPlayoutEnabled(!deaf)
        _state.update { it.copy(deafened = deaf) }
    }

    fun newTask() { _state.update { it.copy(task = null, phase = "idle") } }

    fun capturePhoto() {
        rtc?.send(Capture(kind = "photo", id = "c${System.currentTimeMillis()}", ts = System.currentTimeMillis()))
    }

    fun uiEvent(event: String, target: String?, data: JsonElement?) { rtc?.send(UiEvent(event, target, data)) }
    fun expandPage(page: PageRender?) { _state.update { it.copy(expandedPage = page) } }

    // --- connection lifecycle ---

    private suspend fun connectLoop() {
        var backoffMs = 1000L
        while (true) {
            val ended = CompletableDeferred<String>()
            val client = RtcClient(getApplication(), serverUrl, eglBase, object : RtcClient.Listener {
                override fun onMessage(msg: Message) = handle(msg)
                override fun onConnectionState(state: String) {
                    _state.update { it.copy(connection = state) }
                    if (state in setOf("failed", "closed", "disconnected")) ended.complete(state)
                }
            })
            rtc = client
            try {
                _state.update { it.copy(connection = "connecting") }
                client.connect(sessionId = _state.value.sessionId)
                client.addSink(previewSink)
                client.setMicEnabled(!_state.value.micMuted)   // keep the choices across reconnects
                client.setPlayoutEnabled(!_state.value.deafened)
                tts.routeToSpeaker()                            // WebRTC re-inits audio on connect; re-pin the route
                backoffMs = 1000L
                launch { onConnected(client) }
                val why = ended.await()
                Log.w("Session", "connection $why")
            } catch (e: Exception) {
                Log.e("Session", "connect failed: ${e.message}")
                _state.update { it.copy(connection = "offline") }
            } finally {
                rtc = null
                client.close()
            }
            delay(backoffMs)
            backoffMs = (backoffMs * 2).coerceAtMost(15_000)
        }
    }

    private suspend fun onConnected(client: RtcClient) {
        repeat(30) { if (client.channelOpen) return@repeat else delay(100) }   // channel opens just after the peer
        client.send(SessionStart(session_id = _state.value.sessionId))
        _state.value.task?.let { client.send(TaskSet(it)) }                    // re-arm the task after a reconnect
        var lastBytes = 0L; var lastT = System.currentTimeMillis()
        while (rtc === client) {
            client.send(Ping(t = System.currentTimeMillis()))
            kotlinx.coroutines.withContext(kotlinx.coroutines.Dispatchers.Default) { client.stats() }?.let { s ->
                val now = System.currentTimeMillis()
                val kbps = (s.bytesSent - lastBytes) * 8.0 / (now - lastT).coerceAtLeast(1)
                lastBytes = s.bytesSent; lastT = now
                _state.update { it.copy(stats = s, kbps = kbps) }
            }
            delay(1000)
        }
    }

    private fun launch(block: suspend () -> Unit) = viewModelScope.launch { block() }

    // --- server -> state ---

    private fun handle(msg: Message) {
        when (msg) {
            is SessionState -> {
                prefs.edit().putString("session_id", msg.session_id).apply()   // same identity next launch
                _state.update { it.copy(sessionId = msg.session_id, task = msg.task ?: it.task) }
            }
            is Status -> _state.update { it.copy(phase = msg.phase, statusText = msg.text) }
            is AgentMessage -> append(ChatItem(nextId++, "agent", msg.text, msg.status, msg.url, latencyMs = msg.latency_ms))
            is PageRender -> append(ChatItem(nextId++, "agent", "", status = "page", page = msg))
            is CaptureRequest -> _state.update { it.copy(hint = msg.hint) }
            is CaptureAck -> _state.update {
                val item = msg.url?.let { u -> CaptureItem(msg.id, u, msg.frames, System.currentTimeMillis()) }
                it.copy(hint = null, captures = if (item != null) it.captures + item else it.captures)
            }
            is Pong -> _state.update { it.copy(rttMs = System.currentTimeMillis() - msg.t) }
            is OverlaySet -> _state.update { it.copy(markers = msg.markers, frameW = msg.frame_w, frameH = msg.frame_h) }
            is Transcript -> append(ChatItem(nextId++, "user", msg.text, status = "voice"))
            is TtsChunk -> tts.play(msg)
            else -> Log.d("Session", "unhandled $msg")
        }
    }

    private fun append(item: ChatItem) = _state.update { it.copy(messages = (it.messages + item).takeLast(500)) }

    override fun onCleared() { rtc?.close(); tts.release(); eglBase.release() }
}

/** A VideoSink the UI can attach to once, forwarding to whichever track is live. */
class ProxySink : VideoSink {
    @Volatile var target: VideoSink? = null
    override fun onFrame(frame: VideoFrame) { target?.onFrame(frame) }
}
