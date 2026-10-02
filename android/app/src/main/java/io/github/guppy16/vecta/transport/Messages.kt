package io.github.guppy16.vecta.transport

import kotlinx.serialization.ExperimentalSerializationApi
import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonClassDiscriminator
import kotlinx.serialization.json.JsonElement

/**
 * Control-channel messages, mirroring `agent/vecta/protocol/messages.py`.
 * One JSON object per message; the `type` field selects the subclass.
 */
@OptIn(ExperimentalSerializationApi::class)
@Serializable
@JsonClassDiscriminator("type")
sealed class Message

// --- phone -> server ---

@Serializable @SerialName("session.start")
data class SessionStart(val session_id: String? = null, val device: Map<String, Int> = emptyMap()) : Message()

@Serializable @SerialName("task.set")
data class TaskSet(val text: String) : Message()

@Serializable @SerialName("input.text")
data class InputText(val text: String) : Message()

@Serializable @SerialName("ui.event")
data class UiEvent(val event: String, val target: String? = null, val data: JsonElement? = null) : Message()

@Serializable @SerialName("capture")
data class Capture(val kind: String, val id: String, val ts: Long) : Message()

@Serializable @SerialName("ping")
data class Ping(val t: Long) : Message()

// --- server -> phone ---

@Serializable @SerialName("session.state")
data class SessionState(val session_id: String, val task: String? = null) : Message()

@Serializable @SerialName("status")
data class Status(val phase: String, val text: String = "") : Message()

@Serializable @SerialName("page.render")
data class PageRender(val html: String, val version: Int) : Message()

@Serializable @SerialName("capture.request")
data class CaptureRequest(val kind: String, val hint: String = "") : Message()

@Serializable @SerialName("capture.ack")
data class CaptureAck(val id: String, val frames: Int, val url: String? = null) : Message()

@Serializable @SerialName("agent.message")
data class AgentMessage(
    val text: String,
    val status: String = "info",       // searching | found | info | answer
    val url: String? = null,
    val latency_ms: Int? = null,
) : Message()

@Serializable
data class MarkerSpec(
    val id: String, val label: String = "", val x: Float, val y: Float,
    val w: Float = 0f, val h: Float = 0f, val color: String = "#46C46A",
)

@Serializable @SerialName("overlay.set")
data class OverlaySet(val markers: List<MarkerSpec> = emptyList(), val frame_w: Int = 0, val frame_h: Int = 0) : Message()

@Serializable @SerialName("transcript")
data class Transcript(val text: String, val final: Boolean = true) : Message()

@Serializable @SerialName("tts.chunk")
data class TtsChunk(val id: String, val seq: Int, val last: Boolean, val data: String) : Message()

@Serializable @SerialName("pong")
data class Pong(val t: Long, val server_t: Long) : Message()

object Protocol {
    private val json = Json {
        ignoreUnknownKeys = true      // either side may add fields
        encodeDefaults = true
        explicitNulls = false
    }

    fun encode(msg: Message): String = json.encodeToString(Message.serializer(), msg)

    /** Returns null (and logs) for malformed or unknown messages rather than crashing. */
    fun decode(text: String): Message? = runCatching {
        json.decodeFromString(Message.serializer(), text)
    }.onFailure { android.util.Log.w("Protocol", "bad message: ${it.message}") }.getOrNull()
}
