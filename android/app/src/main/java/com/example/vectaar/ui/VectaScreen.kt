package com.example.vectaar.ui

import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.gestures.detectTransformGestures
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.aspectRatio
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.imePadding
import androidx.compose.foundation.layout.navigationBarsPadding
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.statusBarsPadding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.grid.GridCells
import androidx.compose.foundation.lazy.grid.LazyVerticalGrid
import androidx.compose.foundation.lazy.grid.items
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.BasicTextField
import androidx.compose.foundation.text.KeyboardActions
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.darkColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableFloatStateOf
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.SolidColor
import androidx.compose.ui.graphics.graphicsLayer
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.input.ImeAction
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.compose.ui.window.Dialog
import androidx.compose.ui.window.DialogProperties
import coil3.compose.AsyncImage
import com.example.vectaar.CaptureItem
import com.example.vectaar.ChatItem
import com.example.vectaar.SessionViewModel
import com.example.vectaar.UiState
import com.example.vectaar.transport.PageRender
import org.webrtc.RendererCommon
import org.webrtc.SurfaceViewRenderer

private val TABS = listOf("Comments", "Captures", "Stats")
private const val CAMERA_ASPECT = 4f / 3f   // width : height of the viewfinder

/**
 * Livestream-style shell: video on top, one title/stats line, a compact
 * comment stream (the agent's messages), and a slim input at the bottom.
 */
@Composable
fun VectaScreen(vm: SessionViewModel) {
    val ui by vm.state.collectAsState()
    var tab by remember { mutableIntStateOf(0) }
    var zoomUrl by remember { mutableStateOf<String?>(null) }

    MaterialTheme(colorScheme = darkColorScheme()) {   // always dark: it's a camera app
        Surface(Modifier.fillMaxSize(), color = MaterialTheme.colorScheme.background) {
            Column(Modifier.fillMaxSize().statusBarsPadding().navigationBarsPadding().imePadding()) {
                Viewfinder(vm, ui)
                TitleLine(ui, onClearTask = vm::newTask)
                TabChips(tab) { tab = it }
                Box(Modifier.weight(1f)) {
                    when (tab) {
                        0 -> CommentStream(ui.messages, onImage = { zoomUrl = it }, onPage = vm::expandPage)
                        1 -> CapturesGrid(ui.captures, onImage = { zoomUrl = it })
                        else -> StatsList(ui)
                    }
                }
                InputLine(ui, onSubmit = vm::submit, onShutter = vm::capturePhoto)
            }
        }
        zoomUrl?.let { ZoomDialog(it) { zoomUrl = null } }
        ui.expandedPage?.let { page ->
            PageSheet(page, baseUrl = vm.serverUrl, onDismiss = { vm.expandPage(null) }, onEvent = vm::uiEvent)
        }
    }
}

@Composable
private fun Viewfinder(vm: SessionViewModel, ui: UiState) {
    Box(Modifier.fillMaxWidth().aspectRatio(CAMERA_ASPECT).background(Color.Black)) {
        AndroidView(
            modifier = Modifier.fillMaxSize(),
            factory = { ctx ->
                SurfaceViewRenderer(ctx).apply {
                    init(vm.eglBase.eglBaseContext, null)
                    setScalingType(RendererCommon.ScalingType.SCALE_ASPECT_FILL)
                    setEnableHardwareScaler(true)
                    vm.previewSink.target = this
                }
            },
        )
        ui.hint?.let {
            Text(it, color = Color.White, fontSize = 14.sp, modifier = Modifier.align(Alignment.Center)
                .background(Color.Black.copy(alpha = .6f), RoundedCornerShape(8.dp)).padding(8.dp))
        }
        val dot = when (ui.connection) { "connected" -> Color(0xFF46C46A); "connecting" -> Color(0xFFE0A83C); else -> Color(0xFFE0453C) }
        Row(Modifier.align(Alignment.TopEnd).padding(6.dp).background(Color.Black.copy(alpha = .5f), RoundedCornerShape(10.dp))
            .padding(horizontal = 6.dp, vertical = 3.dp), verticalAlignment = Alignment.CenterVertically) {
            Box(Modifier.size(6.dp).clip(CircleShape).background(dot))
            Text("  ${ui.stats?.fps?.toInt() ?: "–"} fps", color = Color.White, fontSize = 10.sp)
        }
    }
}

/** One line: the task (the stream "title") and the live numbers. */
@Composable
private fun TitleLine(ui: UiState, onClearTask: () -> Unit) {
    Row(Modifier.fillMaxWidth().padding(horizontal = 10.dp, vertical = 6.dp), verticalAlignment = Alignment.CenterVertically) {
        Text(ui.task ?: "no task yet", fontSize = 13.sp, fontWeight = FontWeight.SemiBold, maxLines = 1,
            modifier = Modifier.weight(1f), color = if (ui.task == null) MaterialTheme.colorScheme.onSurfaceVariant else MaterialTheme.colorScheme.onSurface)
        if (ui.task != null) Text("  ✕  ", fontSize = 13.sp, color = MaterialTheme.colorScheme.onSurfaceVariant,
            modifier = Modifier.clickable(onClick = onClearTask))
        val latency = ui.messages.lastOrNull { it.role == "agent" && it.latencyMs != null }?.latencyMs
        val bits = listOfNotNull(ui.rttMs?.let { "$it ms" }, latency?.let { "vlm ${it} ms" }, ui.phase)
        Text(bits.joinToString(" · "), fontSize = 11.sp, fontFamily = FontFamily.Monospace, color = MaterialTheme.colorScheme.onSurfaceVariant)
    }
}

@Composable
private fun TabChips(selected: Int, onSelect: (Int) -> Unit) {
    Row(Modifier.fillMaxWidth().padding(horizontal = 10.dp), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
        TABS.forEachIndexed { i, t ->
            val on = i == selected
            Text(t, fontSize = 12.sp, fontWeight = if (on) FontWeight.SemiBold else FontWeight.Normal,
                color = if (on) MaterialTheme.colorScheme.onPrimaryContainer else MaterialTheme.colorScheme.onSurfaceVariant,
                modifier = Modifier.clip(RoundedCornerShape(12.dp))
                    .background(if (on) MaterialTheme.colorScheme.primaryContainer else Color.Transparent)
                    .clickable { onSelect(i) }.padding(horizontal = 10.dp, vertical = 4.dp))
        }
    }
}

/** Compact rows, newest at the bottom, like a live chat. */
@Composable
private fun CommentStream(items: List<ChatItem>, onImage: (String) -> Unit, onPage: (PageRender) -> Unit) {
    val listState = rememberLazyListState()
    LaunchedEffect(items.size) { if (items.isNotEmpty()) listState.animateScrollToItem(items.size - 1) }
    LazyColumn(Modifier.fillMaxSize(), state = listState, contentPadding = PaddingValues(horizontal = 10.dp, vertical = 4.dp),
        verticalArrangement = Arrangement.spacedBy(2.dp)) {
        if (items.isEmpty()) item {
            Text("Type a task below — e.g. “tell me when a mug appears”.", fontSize = 13.sp,
                color = MaterialTheme.colorScheme.onSurfaceVariant, modifier = Modifier.padding(vertical = 8.dp))
        }
        items(items, key = { it.id }) { CommentRow(it, onImage, onPage) }
    }
}

@Composable
private fun CommentRow(item: ChatItem, onImage: (String) -> Unit, onPage: (PageRender) -> Unit) {
    val muted = MaterialTheme.colorScheme.onSurfaceVariant
    Row(Modifier.fillMaxWidth().padding(vertical = 3.dp), verticalAlignment = Alignment.CenterVertically) {
        Column(Modifier.weight(1f)) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                if (item.role == "user") {
                    Text(if (item.status == "task") "task" else "you", fontSize = 11.sp, fontWeight = FontWeight.SemiBold, color = MaterialTheme.colorScheme.primary)
                } else {
                    Box(Modifier.size(7.dp).clip(CircleShape).background(statusColor(item.status)))
                    Text("  ${item.status}", fontSize = 11.sp, fontWeight = FontWeight.SemiBold, color = statusColor(item.status))
                    item.latencyMs?.let { Text("  ${it} ms", fontSize = 10.sp, color = muted) }
                }
            }
            if (item.text.isNotEmpty()) Text(item.text, fontSize = 13.sp, lineHeight = 17.sp)
            item.page?.let { Text("page v${it.version} — open", fontSize = 13.sp, color = MaterialTheme.colorScheme.primary,
                modifier = Modifier.clickable { onPage(it) }) }
        }
        item.imageUrl?.let {
            AsyncImage(model = it, contentDescription = null, modifier = Modifier.padding(start = 8.dp).size(52.dp)
                .clip(RoundedCornerShape(6.dp)).background(Color.Black).clickable { onImage(it) })
        }
    }
}

fun statusColor(status: String) = when (status) {
    "found" -> Color(0xFF46C46A); "info" -> Color(0xFFE0A83C); "answer" -> Color(0xFF5AA9E6); else -> Color(0xFF8A8F98)
}

@Composable
private fun CapturesGrid(captures: List<CaptureItem>, onImage: (String) -> Unit) {
    if (captures.isEmpty()) return Box(Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
        Text("No captures yet — tap the shutter.", fontSize = 13.sp, color = MaterialTheme.colorScheme.onSurfaceVariant)
    }
    LazyVerticalGrid(GridCells.Fixed(4), Modifier.fillMaxSize(), contentPadding = PaddingValues(8.dp),
        horizontalArrangement = Arrangement.spacedBy(4.dp), verticalArrangement = Arrangement.spacedBy(4.dp)) {
        items(captures.asReversed(), key = { it.id }) { c ->
            AsyncImage(model = c.url, contentDescription = null, modifier = Modifier.aspectRatio(1f)
                .clip(RoundedCornerShape(6.dp)).background(Color.Black).clickable { onImage(c.url) })
        }
    }
}

@Composable
private fun StatsList(ui: UiState) {
    val s = ui.stats
    val rows = listOf(
        "connection" to ui.connection,
        "session" to (ui.sessionId ?: "–"),
        "control RTT" to (ui.rttMs?.let { "$it ms" } ?: "–"),
        "ICE RTT" to (s?.rttMs?.let { "%.0f ms".format(it) } ?: "–"),
        "encoder" to (s?.let { "${it.width ?: "?"}×${it.height ?: "?"} ${it.codec ?: ""}" } ?: "–"),
        "fps" to (s?.fps?.let { "%.1f".format(it) } ?: "–"),
        "bitrate" to (ui.kbps?.let { "%.0f kbps".format(it) } ?: "–"),
        "available" to (s?.availableKbps?.let { "%.0f kbps".format(it) } ?: "–"),
        "sent" to (s?.let { "%.1f MB".format(it.bytesSent / 1e6) } ?: "–"),
        "frames encoded" to (s?.framesEncoded?.toString() ?: "–"),
        "phase" to ui.phase,
    )
    Column(Modifier.fillMaxSize().padding(horizontal = 12.dp, vertical = 6.dp), verticalArrangement = Arrangement.spacedBy(3.dp)) {
        rows.forEach { (k, v) ->
            Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
                Text(k, fontSize = 12.sp, color = MaterialTheme.colorScheme.onSurfaceVariant)
                Text(v, fontSize = 12.sp, fontFamily = FontFamily.Monospace)
            }
        }
    }
}

/** Single-line input; Enter (or the arrow) sends. First message sets the task. */
@Composable
private fun InputLine(ui: UiState, onSubmit: (String) -> Unit, onShutter: () -> Unit) {
    var text by remember { mutableStateOf("") }
    val send = { if (text.isNotBlank()) { onSubmit(text); text = "" } }
    Row(Modifier.fillMaxWidth().padding(horizontal = 8.dp, vertical = 6.dp), verticalAlignment = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.spacedBy(8.dp)) {
        Box(Modifier.size(34.dp).clip(CircleShape).background(Color.White.copy(alpha = .9f)).clickable(onClick = onShutter),
            contentAlignment = Alignment.Center) { Box(Modifier.size(26.dp).clip(CircleShape).background(Color.White)) }
        Box(Modifier.weight(1f).height(38.dp).clip(RoundedCornerShape(19.dp)).background(MaterialTheme.colorScheme.surfaceVariant)
            .padding(horizontal = 14.dp), contentAlignment = Alignment.CenterStart) {
            if (text.isEmpty()) Text(if (ui.task == null) "What should I watch for?" else "Ask about what I’ve seen…",
                fontSize = 14.sp, color = MaterialTheme.colorScheme.onSurfaceVariant)
            BasicTextField(value = text, onValueChange = { text = it }, singleLine = true,
                textStyle = TextStyle(color = MaterialTheme.colorScheme.onSurface, fontSize = 14.sp),
                cursorBrush = SolidColor(MaterialTheme.colorScheme.primary),
                keyboardOptions = KeyboardOptions(imeAction = ImeAction.Send),
                keyboardActions = KeyboardActions(onSend = { send() }), modifier = Modifier.fillMaxWidth())
        }
        Text("➤", fontSize = 20.sp, color = if (text.isBlank()) MaterialTheme.colorScheme.onSurfaceVariant else MaterialTheme.colorScheme.primary,
            modifier = Modifier.clickable(onClick = send).padding(4.dp))
    }
}

/** Full-screen image with pinch-zoom and pan; tap outside the image to close. */
@Composable
private fun ZoomDialog(url: String, onClose: () -> Unit) {
    var scale by remember { mutableFloatStateOf(1f) }
    var offset by remember { mutableStateOf(Offset.Zero) }
    Dialog(onDismissRequest = onClose, properties = DialogProperties(usePlatformDefaultWidth = false)) {
        Box(Modifier.fillMaxSize().background(Color.Black).clickable(onClick = onClose)
            .pointerInput(Unit) {
                detectTransformGestures { _, pan, zoom, _ ->
                    scale = (scale * zoom).coerceIn(1f, 6f)
                    offset = if (scale == 1f) Offset.Zero else offset + pan
                }
            }) {
            AsyncImage(model = url, contentDescription = null, modifier = Modifier.fillMaxSize()
                .graphicsLayer(scaleX = scale, scaleY = scale, translationX = offset.x, translationY = offset.y))
        }
    }
}
