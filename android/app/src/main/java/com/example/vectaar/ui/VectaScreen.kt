package com.example.vectaar.ui

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.border
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
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.BasicTextField
import androidx.compose.foundation.text.KeyboardActions
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.Send
import androidx.compose.material.icons.filled.Close
import androidx.compose.material.icons.outlined.BarChart
import androidx.compose.material.icons.outlined.PhotoLibrary
import androidx.compose.material.icons.automirrored.outlined.Chat
import androidx.compose.material.icons.automirrored.filled.VolumeUp
import androidx.compose.material.icons.automirrored.filled.VolumeOff
import androidx.compose.material.icons.filled.MicOff
import androidx.compose.material.icons.filled.Mic
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.IconButtonDefaults
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
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.graphics.vector.addPathNodes
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.input.ImeAction
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.compose.ui.text.drawText
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

private val TAB_ICONS = listOf(
    Icons.AutoMirrored.Outlined.Chat to "Comments",
    Icons.Outlined.PhotoLibrary to "Captures",
    Icons.Outlined.BarChart to "Stats",
)
private const val CAMERA_ASPECT = 4f / 3f   // width : height of the viewfinder

private val Green = Color(0xFF4CC38A)
private val Amber = Color(0xFFE5A83C)
private val Blue = Color(0xFF5AA9E6)
private val Red = Color(0xFFE5484D)
private val Grey = Color(0xFF8A8F98)

/**
 * Livestream-style shell: video on top, a compact header (task + live numbers +
 * pane chips), the selected pane, and one input bar at the bottom. Dark only.
 */
@Composable
fun VectaScreen(vm: SessionViewModel) {
    val ui by vm.state.collectAsState()
    var tab by remember { mutableIntStateOf(0) }
    var zoomUrl by remember { mutableStateOf<String?>(null) }

    MaterialTheme(colorScheme = darkColorScheme()) {   // always dark: it's a camera app
        Surface(Modifier.fillMaxSize(), color = MaterialTheme.colorScheme.surface) {
            Column(Modifier.fillMaxSize().statusBarsPadding().navigationBarsPadding().imePadding()) {
                Viewfinder(vm, ui)
                Header(ui, tab, onSelectTab = { tab = it }, onClearTask = vm::newTask, onToggleMic = vm::toggleMic, onToggleDeafen = vm::toggleDeafen)
                Box(Modifier.weight(1f)) {
                    when (tab) {
                        0 -> CommentStream(ui.messages, onImage = { zoomUrl = it }, onPage = vm::expandPage)
                        1 -> CapturesGrid(ui.captures, onImage = { zoomUrl = it })
                        else -> StatsList(ui)
                    }
                }
                InputBar(ui, onSubmit = vm::submit, onShutter = vm::capturePhoto)
            }
        }
        zoomUrl?.let { ZoomDialog(it) { zoomUrl = null } }
        ui.expandedPage?.let { page ->
            PageSheet(page, baseUrl = vm.serverUrl, onDismiss = { vm.expandPage(null) }, onEvent = vm::uiEvent)
        }
    }
}

// --- viewfinder ---

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
        MarkerOverlay(ui, Modifier.fillMaxSize())
        ui.hint?.let {
            Text(it, style = MaterialTheme.typography.bodyMedium, color = Color.White,
                modifier = Modifier.align(Alignment.Center).overlay().padding(horizontal = 12.dp, vertical = 8.dp))
        }
        ConnectionChip(ui, Modifier.align(Alignment.TopEnd).padding(8.dp))
    }
}

/**
 * Draws the agent's markers over the live view. Marker coords are normalised to the
 * streamed frame; the renderer shows that frame SCALE_ASPECT_FILL-cropped, so map through
 * the same scale/offset here.
 */
@Composable
private fun MarkerOverlay(ui: UiState, modifier: Modifier) {
    if (ui.markers.isEmpty() || ui.frameW == 0 || ui.frameH == 0) return
    val textMeasurer = androidx.compose.ui.text.rememberTextMeasurer()
    val labelStyle = MaterialTheme.typography.labelLarge.copy(color = Color.White)
    androidx.compose.foundation.Canvas(modifier) {
        val scale = maxOf(size.width / ui.frameW, size.height / ui.frameH)
        val offX = (size.width - ui.frameW * scale) / 2f
        val offY = (size.height - ui.frameH * scale) / 2f
        for (mk in ui.markers) {
            val color = runCatching { Color(android.graphics.Color.parseColor(mk.color)) }.getOrDefault(Color.Green)
            val cx = offX + mk.x * ui.frameW * scale
            val cy = offY + mk.y * ui.frameH * scale
            val w = mk.w * ui.frameW * scale
            val h = mk.h * ui.frameH * scale
            if (w > 8f && h > 8f) {
                drawRoundRect(color, topLeft = androidx.compose.ui.geometry.Offset(cx - w / 2, cy - h / 2),
                    size = androidx.compose.ui.geometry.Size(w, h), cornerRadius = androidx.compose.ui.geometry.CornerRadius(12f),
                    style = androidx.compose.ui.graphics.drawscope.Stroke(width = 6f))
            } else {
                drawCircle(color, radius = 28f, center = androidx.compose.ui.geometry.Offset(cx, cy),
                    style = androidx.compose.ui.graphics.drawscope.Stroke(width = 6f))
                drawCircle(color, radius = 6f, center = androidx.compose.ui.geometry.Offset(cx, cy))
            }
            if (mk.label.isNotEmpty()) {
                val layout = textMeasurer.measure(mk.label, labelStyle)
                val pad = 10f
                val left = cx - w / 2
                val top = cy - h / 2 - layout.size.height - 2 * pad - 4f
                drawRoundRect(color, topLeft = androidx.compose.ui.geometry.Offset(left, top),
                    size = androidx.compose.ui.geometry.Size(layout.size.width + 2 * pad, layout.size.height + 2 * pad),
                    cornerRadius = androidx.compose.ui.geometry.CornerRadius(8f))
                drawText(layout, topLeft = androidx.compose.ui.geometry.Offset(left + pad, top + pad))
            }
        }
    }
}

/** Connected: a green dot and the encoder fps. Otherwise the connection state, in words. */
@Composable
private fun ConnectionChip(ui: UiState, modifier: Modifier) {
    val (dot, label) = when (ui.connection) {
        "connected" -> Green to "${ui.stats?.fps?.toInt() ?: "–"} fps"
        "connecting" -> Amber to if (ui.sessionId == null) "connecting…" else "reconnecting…"
        else -> Red to "offline"
    }
    Row(modifier.overlay().padding(horizontal = 8.dp, vertical = 4.dp),
        verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(6.dp)) {
        Box(Modifier.size(7.dp).clip(CircleShape).background(dot))
        Text(label, style = MaterialTheme.typography.labelSmall, color = Color.White)
    }
}

/** Translucent pill drawn over the video (not a Material surface: it sits on live camera pixels). */
private fun Modifier.overlay() = background(Color.Black.copy(alpha = .55f), RoundedCornerShape(12.dp))

// --- header: task line + pane chips, one block ---

@Composable
private fun Header(ui: UiState, tab: Int, onSelectTab: (Int) -> Unit, onClearTask: () -> Unit,
                   onToggleMic: () -> Unit, onToggleDeafen: () -> Unit) {
    val muted = MaterialTheme.colorScheme.onSurfaceVariant
    Column(Modifier.fillMaxWidth().background(MaterialTheme.colorScheme.surfaceContainer)
        .padding(horizontal = 12.dp, vertical = 8.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
        Row(Modifier.fillMaxWidth().height(40.dp), verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            Row(Modifier.weight(1f), verticalAlignment = Alignment.CenterVertically) {
                Text(ui.task ?: "No task yet", style = MaterialTheme.typography.titleSmall, maxLines = 1,
                    overflow = TextOverflow.Ellipsis, modifier = Modifier.weight(1f, fill = false),
                    color = if (ui.task == null) muted else MaterialTheme.colorScheme.onSurface)
                if (ui.task != null) IconButton(onClick = onClearTask, modifier = Modifier.size(40.dp)) {
                    Icon(Icons.Filled.Close, contentDescription = "Clear task", tint = muted, modifier = Modifier.size(20.dp))
                }
            }
            IconButton(onClick = onToggleMic, modifier = Modifier.size(40.dp)) {
                Icon(if (ui.micMuted) Icons.Filled.MicOff else Icons.Filled.Mic, contentDescription = "Mute microphone",
                    tint = if (ui.micMuted) MaterialTheme.colorScheme.error else muted, modifier = Modifier.size(22.dp))
            }
            IconButton(onClick = onToggleDeafen, modifier = Modifier.size(40.dp)) {
                Icon(if (ui.deafened) Icons.AutoMirrored.Filled.VolumeOff else Icons.AutoMirrored.Filled.VolumeUp,
                    contentDescription = "Mute agent voice",
                    tint = if (ui.deafened) MaterialTheme.colorScheme.error else muted, modifier = Modifier.size(22.dp))
            }
            val latency = ui.messages.lastOrNull { it.role == "agent" && it.latencyMs != null }?.latencyMs
            val bits = listOfNotNull(ui.rttMs?.let { "$it ms" }, latency?.let { "vlm $it ms" },
                ui.phase.takeIf { it != "idle" })
            Text(bits.joinToString(" · "), style = MaterialTheme.typography.labelSmall, fontFamily = FontFamily.Monospace,
                color = muted, maxLines = 1)
        }
        Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(4.dp), verticalAlignment = Alignment.CenterVertically) {
            TAB_ICONS.forEachIndexed { i, (icon, label) ->
                val on = i == tab
                IconButton(onClick = { onSelectTab(i) }, modifier = Modifier.size(40.dp)
                    .clip(RoundedCornerShape(12.dp))
                    .background(if (on) MaterialTheme.colorScheme.secondaryContainer else Color.Transparent)) {
                    Icon(icon, contentDescription = label, modifier = Modifier.size(22.dp),
                        tint = if (on) MaterialTheme.colorScheme.onSecondaryContainer else muted)
                }
            }
        }
    }
}

// --- panes ---

@Composable
private fun EmptyState(text: String) {
    Box(Modifier.fillMaxSize().padding(24.dp), contentAlignment = Alignment.Center) {
        Text(text, style = MaterialTheme.typography.bodyMedium, color = MaterialTheme.colorScheme.onSurfaceVariant,
            textAlign = TextAlign.Center)
    }
}

/** Compact rows, newest at the bottom, like a live chat. */
@Composable
private fun CommentStream(items: List<ChatItem>, onImage: (String) -> Unit, onPage: (PageRender) -> Unit) {
    if (items.isEmpty()) {
        EmptyState("No comments yet.\nType a task below, e.g. “tell me when a mug appears”.")
        return
    }
    val listState = rememberLazyListState()
    LaunchedEffect(items.size) { listState.animateScrollToItem(items.size - 1) }
    LazyColumn(Modifier.fillMaxSize(), state = listState, contentPadding = PaddingValues(horizontal = 12.dp, vertical = 8.dp),
        verticalArrangement = Arrangement.spacedBy(4.dp)) {
        items(items, key = { it.id }) { CommentRow(it, onImage, onPage) }
    }
}

/** The user's rows sit on a raised tone; the agent's rows are flat with a status dot + word. */
@Composable
private fun CommentRow(item: ChatItem, onImage: (String) -> Unit, onPage: (PageRender) -> Unit) {
    val mine = item.role == "user"
    val muted = MaterialTheme.colorScheme.onSurfaceVariant
    val rowShape = if (mine) Modifier.clip(RoundedCornerShape(8.dp)).background(MaterialTheme.colorScheme.surfaceContainerHigh)
        .padding(horizontal = 10.dp) else Modifier
    Row(Modifier.fillMaxWidth().then(rowShape).padding(vertical = 6.dp), verticalAlignment = Alignment.Top,
        horizontalArrangement = Arrangement.spacedBy(10.dp)) {
        Column(Modifier.weight(1f), verticalArrangement = Arrangement.spacedBy(2.dp)) {
            Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                if (mine) {
                    Text(when (item.status) { "task" -> "task"; "voice" -> "you (voice)"; else -> "you" }, style = MaterialTheme.typography.labelSmall,
                        fontWeight = FontWeight.SemiBold, color = MaterialTheme.colorScheme.primary)
                } else {
                    Box(Modifier.size(7.dp).clip(CircleShape).background(statusColor(item.status)))
                    Text(item.status, style = MaterialTheme.typography.labelSmall, fontWeight = FontWeight.SemiBold,
                        color = statusColor(item.status))
                    item.latencyMs?.let {
                        Text("$it ms", style = MaterialTheme.typography.labelSmall, fontFamily = FontFamily.Monospace, color = muted)
                    }
                }
            }
            if (item.text.isNotEmpty()) Text(item.text, style = MaterialTheme.typography.bodyMedium)
            item.page?.let {
                Text("Open page v${it.version}", style = MaterialTheme.typography.bodyMedium,
                    color = MaterialTheme.colorScheme.primary, modifier = Modifier.clickable { onPage(it) })
            }
        }
        item.imageUrl?.let { Thumbnail(it) { onImage(it) } }
    }
}

private fun statusColor(status: String) = when (status) {
    "found" -> Green; "info" -> Amber; "answer" -> Blue; else -> Grey
}

@Composable
private fun Thumbnail(url: String, onClick: () -> Unit) {
    AsyncImage(model = url, contentDescription = null, contentScale = ContentScale.Crop,
        modifier = Modifier.size(52.dp).clip(RoundedCornerShape(8.dp))
            .background(MaterialTheme.colorScheme.surfaceContainerHigh).clickable(onClick = onClick))
}

@Composable
private fun CapturesGrid(captures: List<CaptureItem>, onImage: (String) -> Unit) {
    if (captures.isEmpty()) {
        EmptyState("No captures yet.\nTap the camera button to take one.")
        return
    }
    LazyVerticalGrid(GridCells.Fixed(4), Modifier.fillMaxSize(), contentPadding = PaddingValues(12.dp),
        horizontalArrangement = Arrangement.spacedBy(6.dp), verticalArrangement = Arrangement.spacedBy(6.dp)) {
        items(captures.asReversed(), key = { it.id }) { c ->
            AsyncImage(model = c.url, contentDescription = null, contentScale = ContentScale.Crop,
                modifier = Modifier.aspectRatio(1f).clip(RoundedCornerShape(8.dp))
                    .background(MaterialTheme.colorScheme.surfaceContainerHigh).clickable { onImage(c.url) })
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
    Column(Modifier.fillMaxSize().verticalScroll(rememberScrollState()).padding(horizontal = 12.dp, vertical = 8.dp),
        verticalArrangement = Arrangement.spacedBy(8.dp)) {
        rows.forEach { (k, v) ->
            Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
                Text(k, style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
                Text(v, style = MaterialTheme.typography.bodySmall, fontFamily = FontFamily.Monospace)
            }
        }
    }
}

// --- input ---

/** Shutter · single-line field · send. Enter (IME "send") or the arrow sends; the first message sets the task. */
@Composable
private fun InputBar(ui: UiState, onSubmit: (String) -> Unit, onShutter: () -> Unit) {
    var text by remember { mutableStateOf("") }
    val send = { if (text.isNotBlank()) { onSubmit(text); text = "" } }
    Row(Modifier.fillMaxWidth().background(MaterialTheme.colorScheme.surfaceContainer).padding(horizontal = 4.dp, vertical = 4.dp),
        verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(4.dp)) {
        IconButton(onClick = onShutter) { Icon(CameraIcon, contentDescription = "Take a photo") }
        Box(Modifier.weight(1f).height(40.dp).clip(RoundedCornerShape(20.dp))
            .background(MaterialTheme.colorScheme.surfaceContainerHighest).padding(horizontal = 14.dp),
            contentAlignment = Alignment.CenterStart) {
            if (text.isEmpty()) Text(if (ui.task == null) "What should I watch for?" else "Ask about what I’ve seen…",
                style = MaterialTheme.typography.bodyMedium, color = MaterialTheme.colorScheme.onSurfaceVariant)
            BasicTextField(value = text, onValueChange = { text = it }, singleLine = true,
                textStyle = MaterialTheme.typography.bodyMedium.copy(color = MaterialTheme.colorScheme.onSurface),
                cursorBrush = SolidColor(MaterialTheme.colorScheme.primary),
                keyboardOptions = KeyboardOptions(imeAction = ImeAction.Send),
                keyboardActions = KeyboardActions(onSend = { send() }), modifier = Modifier.fillMaxWidth())
        }
        IconButton(onClick = send, enabled = text.isNotBlank(),
            colors = IconButtonDefaults.iconButtonColors(contentColor = MaterialTheme.colorScheme.primary)) {
            Icon(Icons.AutoMirrored.Filled.Send, contentDescription = "Send")
        }
    }
}

/** Material "photo_camera" (outlined). Inlined: it isn't in material-icons-core, and the extended set is ~20 MB. */
private val CameraIcon: ImageVector by lazy {
    ImageVector.Builder("PhotoCamera", 24.dp, 24.dp, 24f, 24f)
        .addPath(
            addPathNodes(
                "M14.12 4l1.83 2H20v12H4V6h4.05l1.83-2h4.24M15 2H9L7.17 4H4c-1.1 0-2 .9-2 2v12c0 1.1.9 2 2 2h16" +
                    "c1.1 0 2-.9 2-2V6c0-1.1-.9-2-2-2h-3.17L15 2zm-3 7c1.65 0 3 1.35 3 3s-1.35 3-3 3-3-1.35-3-3 1.35-3 3-3" +
                    "m0-2c-2.76 0-5 2.24-5 5s2.24 5 5 5 5-2.24 5-5-2.24-5-5-5z",
            ),
            fill = SolidColor(Color.White),
        )
        .build()
}

// --- image zoom ---

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

