package com.example.vectaar

import androidx.compose.animation.AnimatedVisibility
import androidx.compose.animation.fadeIn
import androidx.compose.animation.fadeOut
import androidx.compose.animation.slideInVertically
import androidx.compose.foundation.background
import androidx.compose.foundation.gestures.detectTapGestures
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.text.input.ImeAction
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import kotlinx.coroutines.delay
import kotlinx.coroutines.withTimeoutOrNull
import org.json.JSONObject

data class ChatMessage(val role: String, val text: String, val meta: String = "")

data class LlmMsg(
    val status: String, val say: String, val latencyMs: Int,
    val promptTokens: Int?, val completionTokens: Int?, val tokPerS: Double?
)

fun parseLlm(json: JSONObject) = LlmMsg(
    status = json.optString("status", "info"),
    say = json.optString("say", ""),
    latencyMs = json.optInt("latency_ms", 0),
    promptTokens = if (json.isNull("prompt_tokens")) null else json.optInt("prompt_tokens"),
    completionTokens = if (json.isNull("completion_tokens")) null else json.optInt("completion_tokens"),
    tokPerS = if (json.isNull("tok_per_s")) null else json.optDouble("tok_per_s")
)

fun statusColor(status: String): Color = when (status) {
    "found" -> Color(0xFF46C46A)
    "info"  -> Color(0xFFE0A83C)
    else    -> Color(0xFF8A8F98)
}

/** Task entry: type the standing task, tap Set to send it to the server. */
@Composable
fun TaskInputBar(currentTask: String, onSet: (String) -> Unit, modifier: Modifier = Modifier) {
    var text by remember { mutableStateOf(currentTask) }
    Row(
        modifier = modifier
            .background(Color.Black.copy(alpha = 0.6f), RoundedCornerShape(10.dp))
            .padding(6.dp),
        verticalAlignment = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.spacedBy(6.dp)
    ) {
        OutlinedTextField(
            value = text, onValueChange = { text = it },
            placeholder = { Text("e.g. find a globe", fontSize = 13.sp) },
            singleLine = true, modifier = Modifier.weight(1f),
            keyboardOptions = androidx.compose.foundation.text.KeyboardOptions(imeAction = ImeAction.Done),
            keyboardActions = androidx.compose.foundation.text.KeyboardActions(onDone = { if (text.isNotBlank()) onSet(text.trim()) }),
            colors = OutlinedTextFieldDefaults.colors(
                focusedTextColor = Color.White, unfocusedTextColor = Color.White,
                focusedBorderColor = Color.Cyan, unfocusedBorderColor = Color.Gray,
                focusedPlaceholderColor = Color.Gray, unfocusedPlaceholderColor = Color.Gray
            )
        )
        Button(onClick = { if (text.isNotBlank()) onSet(text.trim()) }) { Text("Set") }
    }
}

/** One button: quick TAP = snap a single frame, HOLD = record (stream frames). */
@Composable
fun SnapRecordButton(
    recording: Boolean, onSnap: () -> Unit,
    onRecordStart: () -> Unit, onRecordStop: () -> Unit,
    modifier: Modifier = Modifier
) {
    Box(
        modifier = modifier
            .size(76.dp)
            .clip(CircleShape)
            .background(if (recording) Color(0xFFE0453C) else Color.White.copy(alpha = 0.9f))
            .pointerInput(Unit) {
                detectTapGestures(
                    onPress = {
                        val quick = withTimeoutOrNull(300L) { tryAwaitRelease() }
                        if (quick == null) {            // held -> record until release
                            onRecordStart(); tryAwaitRelease(); onRecordStop()
                        } else if (quick) {             // quick tap -> snap
                            onSnap()
                        }
                    }
                )
            },
        contentAlignment = Alignment.Center
    ) {
        Text(if (recording) "REC" else "SNAP",
            color = if (recording) Color.White else Color.Black, fontSize = 13.sp)
    }
}

/** Latest assistant message: floats in, holds ~5s, fades. Tap to open full chat. */
@Composable
fun TransientMessage(latest: ChatMessage?, onExpand: () -> Unit, modifier: Modifier = Modifier) {
    var visible by remember { mutableStateOf(false) }
    LaunchedEffect(latest) {
        if (latest != null && latest.role == "assistant" && latest.text.isNotBlank()) {
            visible = true; delay(5000); visible = false
        } else visible = false
    }
    AnimatedVisibility(
        visible = visible, modifier = modifier,
        enter = fadeIn() + slideInVertically { it / 2 }, exit = fadeOut()
    ) {
        if (latest != null) {
            Row(
                modifier = Modifier
                    .background(Color.Black.copy(alpha = 0.78f), RoundedCornerShape(12.dp))
                    .pointerInput(Unit) { detectTapGestures { onExpand() } }
                    .padding(horizontal = 14.dp, vertical = 10.dp),
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.spacedBy(10.dp)
            ) {
                Box(Modifier.size(9.dp).clip(CircleShape)
                    .background(statusColor(latest.meta.ifBlankOr("found"))))
                Text(latest.text, color = Color.White, fontSize = 16.sp,
                    modifier = Modifier.weight(1f, fill = false))
            }
        }
    }
}

// tiny helper so an empty meta still gets a sensible dot colour
private fun String.ifBlankOr(default: String) = if (this.isBlank()) default else this

/** Full conversation, shown when the chat button is toggled on. */
@Composable
fun ChatSheet(messages: List<ChatMessage>, onClose: () -> Unit, modifier: Modifier = Modifier) {
    Column(
        modifier = modifier
            .fillMaxWidth()
            .fillMaxHeight(0.55f)
            .background(Color.Black.copy(alpha = 0.9f), RoundedCornerShape(topStart = 16.dp, topEnd = 16.dp))
            .padding(12.dp)
    ) {
        Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically) {
            Text("Conversation", color = Color.White, fontSize = 16.sp)
            TextButton(onClick = onClose) { Text("Close", color = Color.Cyan) }
        }
        Spacer(Modifier.height(8.dp))
        LazyColumn(Modifier.weight(1f), verticalArrangement = Arrangement.spacedBy(8.dp)) {
            items(messages) { m ->
                val mine = m.role == "user"
                Column(Modifier.fillMaxWidth(),
                    horizontalAlignment = if (mine) Alignment.End else Alignment.Start) {
                    Text(
                        m.text,
                        color = if (mine) Color.Black else Color.White,
                        fontSize = 15.sp,
                        modifier = Modifier
                            .background(
                                if (mine) Color.Cyan else Color.DarkGray,
                                RoundedCornerShape(10.dp)
                            )
                            .padding(horizontal = 12.dp, vertical = 8.dp)
                    )
                    if (m.meta.isNotBlank())
                        Text(m.meta, color = Color.Gray, fontSize = 10.sp,
                            modifier = Modifier.padding(top = 2.dp, start = 4.dp, end = 4.dp))
                }
            }
        }
    }
}