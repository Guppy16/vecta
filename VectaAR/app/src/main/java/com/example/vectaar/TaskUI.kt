package com.example.vectaar

import android.graphics.BitmapFactory
import androidx.compose.animation.AnimatedVisibility
import androidx.compose.animation.fadeIn
import androidx.compose.animation.fadeOut
import androidx.compose.animation.slideInVertically
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.gestures.detectTapGestures
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.KeyboardActions
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.focus.FocusManager
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalFocusManager
import androidx.compose.ui.platform.LocalSoftwareKeyboardController
import androidx.compose.ui.platform.SoftwareKeyboardController
import androidx.compose.ui.text.input.ImeAction
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import kotlinx.coroutines.delay
import kotlinx.coroutines.withTimeoutOrNull
import org.json.JSONObject

/**
 * role: "user" (text), "model" (a judged frame + verdict), "answer" (Q&A reply).
 * imageBytes = the JPEG the model actually looked at (null for text-only entries).
 */
data class ChatMessage(
    val role: String,
    val text: String,
    val meta: String = "",                       // status for "model" entries
    val imageBytes: ByteArray? = null,
    val point: Pair<Double, Double>? = null
)

data class LlmMsg(
    val status: String, val say: String, val latencyMs: Int,
    val promptTokens: Int?, val completionTokens: Int?, val tokPerS: Double?,
    val ctxTokens: Int?, val ctxMax: Int?, val point: Pair<Double, Double>?
)

fun parseLlm(json: JSONObject) = LlmMsg(
    status = json.optString("status", "info"),
    say = json.optString("say", ""),
    latencyMs = json.optInt("latency_ms", 0),
    promptTokens = if (json.isNull("prompt_tokens")) null else json.optInt("prompt_tokens"),
    completionTokens = if (json.isNull("completion_tokens")) null else json.optInt("completion_tokens"),
    tokPerS = if (json.isNull("tok_per_s")) null else json.optDouble("tok_per_s"),
    ctxTokens = if (json.isNull("ctx_tokens")) null else json.optInt("ctx_tokens"),
    ctxMax = if (json.isNull("ctx_max")) null else json.optInt("ctx_max"),
    point = json.optJSONArray("point")?.let {
        if (it.length() == 2) Pair(it.optDouble(0), it.optDouble(1)) else null
    }
)

fun ctxLine(used: Int?, max: Int?): String {
    if (used == null || max == null || max <= 0) return ""
    fun k(n: Int) = if (n >= 1000) "%.1fk".format(n / 1000.0) else n.toString()
    return "Ctx ${k(used)}/${k(max)} · ${100 * used / max}%"
}

fun statusColor(status: String): Color = when (status) {
    "found" -> Color(0xFF46C46A)
    "info"  -> Color(0xFFE0A83C)
    "answer" -> Color(0xFF5AA9E6)
    else    -> Color(0xFF8A8F98)         // searching
}

private fun dismiss(kb: SoftwareKeyboardController?, fm: FocusManager) { kb?.hide(); fm.clearFocus() }

@Composable
fun TaskInputBar(currentTask: String, onSet: (String) -> Unit, modifier: Modifier = Modifier) {
    var text by remember { mutableStateOf(currentTask) }
    val kb = LocalSoftwareKeyboardController.current
    val fm = LocalFocusManager.current
    fun submit() { if (text.isNotBlank()) { onSet(text.trim()); dismiss(kb, fm) } }
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
            keyboardOptions = KeyboardOptions(imeAction = ImeAction.Done),
            keyboardActions = KeyboardActions(onDone = { submit() }),
            colors = OutlinedTextFieldDefaults.colors(
                focusedTextColor = Color.White, unfocusedTextColor = Color.White,
                focusedBorderColor = Color.Cyan, unfocusedBorderColor = Color.Gray,
                focusedPlaceholderColor = Color.Gray, unfocusedPlaceholderColor = Color.Gray
            )
        )
        Button(onClick = { submit() }) { Text("Set") }
    }
}

@Composable
fun SnapRecordButton(
    recording: Boolean, onSnap: () -> Unit,
    onRecordStart: () -> Unit, onRecordStop: () -> Unit,
    modifier: Modifier = Modifier
) {
    Box(
        modifier = modifier
            .size(76.dp).clip(CircleShape)
            .background(if (recording) Color(0xFFE0453C) else Color.White.copy(alpha = 0.9f))
            .pointerInput(Unit) {
                detectTapGestures(
                    onPress = {
                        val quick = withTimeoutOrNull(300L) { tryAwaitRelease() }
                        if (quick == null) { onRecordStart(); tryAwaitRelease(); onRecordStop() }
                        else if (quick) { onSnap() }
                    }
                )
            },
        contentAlignment = Alignment.Center
    ) {
        Text(if (recording) "REC" else "SNAP",
            color = if (recording) Color.White else Color.Black, fontSize = 13.sp)
    }
}

@Composable
fun TransientMessage(latest: ChatMessage?, onExpand: () -> Unit, modifier: Modifier = Modifier) {
    var visible by remember { mutableStateOf(false) }
    LaunchedEffect(latest) {
        if (latest != null && latest.text.isNotBlank() && latest.role != "user") {
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

private fun String.ifBlankOr(default: String) = if (this.isBlank()) default else this

@Composable
private fun ChatImage(bytes: ByteArray, size: Int, modifier: Modifier = Modifier) {
    val img = remember(bytes) {
        runCatching { BitmapFactory.decodeByteArray(bytes, 0, bytes.size).asImageBitmap() }.getOrNull()
    }
    if (img != null) {
        Image(bitmap = img, contentDescription = null, contentScale = ContentScale.Crop,
            modifier = modifier.size(size.dp).clip(RoundedCornerShape(8.dp)))
    } else {
        Box(modifier.size(size.dp).clip(RoundedCornerShape(8.dp)).background(Color.DarkGray))
    }
}

/** Conversation + a box to ask about what's been seen. Frames are tap-to-expand. */
@Composable
fun ChatSheet(
    messages: List<ChatMessage>, onClose: () -> Unit,
    onAsk: (String) -> Unit, onImageTap: (ByteArray) -> Unit,
    modifier: Modifier = Modifier
) {
    var q by remember { mutableStateOf("") }
    val kb = LocalSoftwareKeyboardController.current
    val fm = LocalFocusManager.current
    fun submit() { if (q.isNotBlank()) { onAsk(q.trim()); q = ""; dismiss(kb, fm) } }
    Column(
        modifier = modifier
            .fillMaxWidth().fillMaxHeight(0.62f)
            .background(Color.Black.copy(alpha = 0.93f), RoundedCornerShape(topStart = 16.dp, topEnd = 16.dp))
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
                if (m.imageBytes != null) {
                    // a frame the model judged + its verdict
                    Row(
                        Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.spacedBy(10.dp)
                    ) {
                        ChatImage(m.imageBytes, size = 60,
                            modifier = Modifier.clickable { onImageTap(m.imageBytes) })
                        Column(Modifier.weight(1f)) {
                            Text(m.meta.ifBlankOr("searching").uppercase(),
                                color = statusColor(m.meta.ifBlankOr("searching")), fontSize = 12.sp)
                            if (m.text.isNotBlank())
                                Text(m.text, color = Color.White, fontSize = 14.sp)
                            if (m.point != null)
                                Text("box @ (%.2f, %.2f)".format(m.point.first, m.point.second),
                                    color = Color.Gray, fontSize = 10.sp)
                        }
                    }
                } else {
                    val mine = m.role == "user"
                    Column(Modifier.fillMaxWidth(),
                        horizontalAlignment = if (mine) Alignment.End else Alignment.Start) {
                        Text(
                            m.text,
                            color = if (mine) Color.Black else Color.White, fontSize = 15.sp,
                            modifier = Modifier
                                .background(if (mine) Color.Cyan else Color.DarkGray, RoundedCornerShape(10.dp))
                                .padding(horizontal = 12.dp, vertical = 8.dp)
                        )
                    }
                }
            }
        }
        Spacer(Modifier.height(8.dp))
        Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(6.dp)) {
            OutlinedTextField(
                value = q, onValueChange = { q = it },
                placeholder = { Text("Ask about what you've seen…", fontSize = 13.sp) },
                singleLine = true, modifier = Modifier.weight(1f),
                keyboardOptions = KeyboardOptions(imeAction = ImeAction.Send),
                keyboardActions = KeyboardActions(onSend = { submit() }),
                colors = OutlinedTextFieldDefaults.colors(
                    focusedTextColor = Color.White, unfocusedTextColor = Color.White,
                    focusedBorderColor = Color.Cyan, unfocusedBorderColor = Color.Gray,
                    focusedPlaceholderColor = Color.Gray, unfocusedPlaceholderColor = Color.Gray
                )
            )
            Button(onClick = { submit() }) { Text("Ask") }
        }
    }
}

/** Full-screen viewer for a sent frame. Tap anywhere to close. */
@Composable
fun ExpandedImage(bytes: ByteArray, onDismiss: () -> Unit) {
    Box(
        Modifier.fillMaxSize().background(Color.Black.copy(alpha = 0.94f))
            .clickable { onDismiss() },
        contentAlignment = Alignment.Center
    ) {
        val img = remember(bytes) {
            runCatching { BitmapFactory.decodeByteArray(bytes, 0, bytes.size).asImageBitmap() }.getOrNull()
        }
        if (img != null)
            Image(bitmap = img, contentDescription = null, contentScale = ContentScale.Fit,
                modifier = Modifier.fillMaxWidth().padding(12.dp))
        Text("tap to close", color = Color.White, fontSize = 12.sp,
            modifier = Modifier.align(Alignment.BottomCenter).padding(28.dp))
    }
}