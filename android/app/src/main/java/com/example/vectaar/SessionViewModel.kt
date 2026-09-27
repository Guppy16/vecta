package com.example.vectaar

import android.app.Application
import android.util.Log
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.example.vectaar.transport.Capture
import com.example.vectaar.transport.CaptureAck
import com.example.vectaar.transport.CaptureRequest
import com.example.vectaar.transport.InputText
import com.example.vectaar.transport.Message
import com.example.vectaar.transport.PageRender
import com.example.vectaar.transport.Ping
import com.example.vectaar.transport.Pong
import com.example.vectaar.transport.RtcClient
import com.example.vectaar.transport.SessionStart
import com.example.vectaar.transport.SessionState
import com.example.vectaar.transport.Status
import com.example.vectaar.transport.TaskSet
import com.example.vectaar.transport.UiEvent
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import kotlinx.serialization.json.JsonElement

/** Everything the UI shows, in one immutable snapshot. */
data class UiState(
    val connection: String = "new",
    val sessionId: String? = null,
    val task: String? = null,
    val phase: String = "idle",
    val statusText: String = "",
    val rttMs: Long? = null,
    val hint: String? = null,               // capture.request hint shown over the camera
    val page: PageRender? = null,
    val lastCapture: CaptureAck? = null,
)

/** Owns the RTC connection and turns server messages into [UiState]. */
class SessionViewModel(app: Application) : AndroidViewModel(app) {
    val serverUrl = "http://${BuildConfig.SERVER_IP}"
    private val _state = MutableStateFlow(UiState())
    val state: StateFlow<UiState> = _state

    val rtc = RtcClient(app, serverUrl, object : RtcClient.Listener {
        override fun onMessage(msg: Message) = handle(msg)
        override fun onConnectionState(state: String) {
            _state.update { it.copy(connection = state) }
            if (state == "connected") viewModelScope.launch { onConnected() }
        }
    })

    init {
        viewModelScope.launch {
            runCatching { rtc.connect() }
                .onFailure { Log.e("Session", "connect failed", it); _state.update { s -> s.copy(connection = "failed: ${it.message}") } }
        }
    }

    private suspend fun onConnected() {
        // the data channel opens slightly after the peer connection does
        repeat(20) { if (rtc.send(SessionStart())) return@repeat else delay(100) }
        while (true) { rtc.send(Ping(t = System.currentTimeMillis())); delay(2000) }
    }

    fun setTask(text: String) { rtc.send(TaskSet(text)); _state.update { it.copy(task = text) } }
    fun sendText(text: String) { rtc.send(InputText(text)) }
    fun uiEvent(event: String, target: String?, data: JsonElement?) { rtc.send(UiEvent(event, target, data)) }
    fun dismissPage() { _state.update { it.copy(page = null) } }

    fun capturePhoto() {
        val id = "c${System.currentTimeMillis()}"
        rtc.send(Capture(kind = "photo", id = id, ts = System.currentTimeMillis()))
    }

    private fun handle(msg: Message) {
        when (msg) {
            is SessionState -> _state.update { it.copy(sessionId = msg.session_id, task = msg.task) }
            is Status -> _state.update { it.copy(phase = msg.phase, statusText = msg.text) }
            is PageRender -> _state.update { it.copy(page = msg) }
            is CaptureRequest -> _state.update { it.copy(hint = msg.hint) }
            is CaptureAck -> _state.update { it.copy(lastCapture = msg, hint = null) }
            is Pong -> _state.update { it.copy(rttMs = System.currentTimeMillis() - msg.t) }
            else -> Log.d("Session", "unhandled $msg")
        }
    }

    override fun onCleared() { rtc.close() }
}
