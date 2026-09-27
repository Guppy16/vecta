package com.example.vectaar.ui

import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.statusBarsPadding
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Button
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.viewinterop.AndroidView
import com.example.vectaar.SessionViewModel
import org.webrtc.RendererCommon
import org.webrtc.SurfaceViewRenderer

/**
 * Layer 1: live camera (the WebRTC track, rendered locally) with a task bar,
 * shutter and status. Layer 2, when the agent has rendered something: [PageSheet].
 */
@Composable
fun CaptureScreen(vm: SessionViewModel) {
    val ui by vm.state.collectAsState()

    Box(Modifier.fillMaxSize().background(Color.Black)) {
        AndroidView(
            modifier = Modifier.fillMaxSize(),
            factory = { ctx ->
                SurfaceViewRenderer(ctx).apply {
                    init(vm.rtc.eglBase.eglBaseContext, null)
                    setScalingType(RendererCommon.ScalingType.SCALE_ASPECT_FILL)
                    setEnableHardwareScaler(true)
                }
            },
        ) { renderer ->
            // (re)attach whenever the composable updates; addSink is idempotent
            vm.rtc.addSink(renderer)
        }

        TaskBar(current = ui.task, onSet = vm::setTask, onText = vm::sendText,
            modifier = Modifier.align(Alignment.TopCenter).statusBarsPadding().padding(12.dp))

        ui.hint?.let {
            Text(it, color = Color.White, fontSize = 16.sp,
                modifier = Modifier.align(Alignment.Center)
                    .background(Color.Black.copy(alpha = .6f), RoundedCornerShape(10.dp)).padding(12.dp))
        }

        Column(
            Modifier.align(Alignment.BottomCenter).padding(bottom = 28.dp),
            horizontalAlignment = Alignment.CenterHorizontally,
            verticalArrangement = Arrangement.spacedBy(10.dp),
        ) {
            Box(
                Modifier.size(76.dp).clip(CircleShape).background(Color.White.copy(alpha = .9f))
                    .clickable { vm.capturePhoto() },
                contentAlignment = Alignment.Center,
            ) { Box(Modifier.size(60.dp).clip(CircleShape).background(Color.White)) }
            val rtt = ui.rttMs?.let { "  ·  ${it} ms" } ?: ""
            val shot = ui.lastCapture?.let { "  ·  shot ${it.frames}f" } ?: ""
            Text("${ui.connection}$rtt  ·  ${ui.phase}$shot", color = Color.Yellow, fontSize = 11.sp)
            if (ui.page != null) Button(onClick = {}) { Text("page v${ui.page!!.version} — swipe up") }
        }
    }

    ui.page?.let { page ->
        PageSheet(page = page, baseUrl = vm.serverUrl, onDismiss = vm::dismissPage, onEvent = vm::uiEvent)
    }
}

@Composable
private fun TaskBar(current: String?, onSet: (String) -> Unit, onText: (String) -> Unit, modifier: Modifier) {
    var text by remember { mutableStateOf("") }
    Row(modifier.fillMaxWidth(), verticalAlignment = Alignment.CenterVertically) {
        OutlinedTextField(
            value = text, onValueChange = { text = it }, singleLine = true,
            placeholder = { Text(current?.let { "task: $it" } ?: "what do you want to do?") },
            modifier = Modifier.weight(1f).background(Color.Black.copy(alpha = .4f), RoundedCornerShape(10.dp)),
        )
        Button(onClick = { if (text.isNotBlank()) { if (current == null) onSet(text) else onText(text); text = "" } },
            modifier = Modifier.padding(start = 8.dp)) { Text(if (current == null) "Set" else "Send") }
    }
}
