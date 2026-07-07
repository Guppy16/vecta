package com.example.vectaar

import android.Manifest
import android.content.pm.PackageManager
import android.os.Bundle
import android.util.Log
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.Text
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import androidx.compose.ui.platform.LocalLifecycleOwner
import kotlinx.coroutines.delay
import okhttp3.*
import okio.ByteString.Companion.toByteString
import org.json.JSONObject
import java.io.IOException
import java.nio.ByteBuffer
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicLong
import kotlin.time.Duration.Companion.seconds

class MainActivity : ComponentActivity() {
    private val requestPermissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { isGranted: Boolean -> if (isGranted) recreate() }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        val hasCameraPermission = ContextCompat.checkSelfPermission(
            this, Manifest.permission.CAMERA
        ) == PackageManager.PERMISSION_GRANTED
        if (!hasCameraPermission) requestPermissionLauncher.launch(Manifest.permission.CAMERA)
        setContent {
            if (hasCameraPermission) CameraScreen()
            else Box(Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
                Text("Camera permission required.")
            }
        }
    }
}

/**
 * Thread-safe bridge between Compose state (main thread) and the CameraX analyzer
 * (background thread). The composable pushes current settings in via @Volatile
 * fields; the analyzer reads them per frame and sends when snap/record asks.
 */
class CaptureController {
    @Volatile var connected = false
    @Volatile var recording = false
    @Volatile var snapRequested = false
    @Volatile var scaleFactor = 4
    @Volatile var intervalMs = 400L
    @Volatile var ws: WebSocket? = null
    @Volatile var onSent: (Int) -> Unit = {}
    private var lastFrame = 0L

    fun analyze(image: ImageProxy) {
        try {
            val now = System.currentTimeMillis()
            val doRecord = connected && recording && (now - lastFrame >= intervalMs)
            val doSnap = connected && snapRequested
            if (doRecord || doSnap) {
                if (doRecord) lastFrame = now
                snapRequested = false
                val yuv = ImageUtils.extractYuv(image, now)
                ImageUtils.yuvToDownsampledJpeg(yuv, scaleFactor)?.let {
                    val b = ByteBuffer.allocate(8 + it.size)
                    b.putLong(now); b.put(it)
                    ws?.send(b.array().toByteString())
                    onSent(8 + it.size)
                }
            }
        } catch (e: Exception) {
            Log.e("Capture", "analyze failed", e)
        } finally {
            image.close()   // ALWAYS release the frame (KEEP_ONLY_LATEST drops the rest)
        }
    }
}

@Composable
fun CameraScreen() {
    // --- task / session state ---
    var task by remember { mutableStateOf("") }
    var connected by remember { mutableStateOf(false) }
    var recording by remember { mutableStateOf(false) }
    var chatMessages by remember { mutableStateOf(listOf<ChatMessage>()) }
    var latestMsg by remember { mutableStateOf<ChatMessage?>(null) }
    var showChat by remember { mutableStateOf(false) }
    var scaleFactor by remember { mutableIntStateOf(4) }
    var status by remember { mutableStateOf("Set a task to begin") }

    // --- telemetry ---
    var llmStats by remember { mutableStateOf("") }
    var pingMs by remember { mutableStateOf("Ping: -- ms") }
    var roundtripLatency by remember { mutableStateOf("RTT: -- ms") }
    var ctxStat by remember { mutableStateOf("") }
    val totalSentBytes = remember { AtomicLong(0L) }
    val totalRecvBytes = remember { AtomicLong(0L) }
    var dataStats by remember { mutableStateOf("Sent 0.0MB  Recv 0.0MB  0 KB/s") }

    val controller = remember { CaptureController() }
    val ip = BuildConfig.SERVER_IP
    val serverUrl = "ws://$ip/ws"
    val pingUrl = "http://$ip/ping"
    val client = remember { OkHttpClient.Builder().readTimeout(3.seconds).build() }
    var webSocket by remember { mutableStateOf<WebSocket?>(null) }

    // send interval: 6400/scaleFactor^2 -> 400ms at 1/4x (constant data rate)
    val baseIntervalMs = 6400L

    // keep the analyzer's view of state current (runs after each recomposition)
    SideEffect {
        controller.connected = connected
        controller.recording = recording
        controller.scaleFactor = scaleFactor
        controller.intervalMs = baseIntervalMs / (scaleFactor.toLong() * scaleFactor.toLong())
        controller.ws = webSocket
        controller.onSent = { n -> totalSentBytes.addAndGet(n.toLong()) }
    }

    val wsListener = remember {
        object : WebSocketListener() {
            override fun onOpen(webSocket: WebSocket, response: Response) { status = "Connected" }

            override fun onMessage(webSocket: WebSocket, text: String) {
                totalRecvBytes.addAndGet(text.length.toLong())
                try {
                    val json = JSONObject(text)
                    when (json.optString("type")) {
                        "llm" -> {                              // detection result
                            val m = parseLlm(json)
                            llmStats = "LLM ${m.latencyMs}ms" + (m.tokPerS?.let { "  $it tok/s" } ?: "")
                            ctxStat = ctxLine(m.ctxTokens, m.ctxMax)
                            if (json.has("ts"))
                                roundtripLatency = "RTT: ${System.currentTimeMillis() - json.getLong("ts")} ms"
                            if (m.say.isNotBlank()) {
                                val cm = ChatMessage("assistant", m.say, m.status)
                                chatMessages = chatMessages + cm; latestMsg = cm
                            }
                        }
                        "answer" -> {                           // reply to a question
                            val txt = json.optString("text", "")
                            llmStats = "LLM ${json.optInt("latency_ms", 0)}ms"
                            val used = if (json.isNull("ctx_tokens")) null else json.optInt("ctx_tokens")
                            val cmax = if (json.isNull("ctx_max")) null else json.optInt("ctx_max")
                            val nkf = json.optInt("n_keyframes", -1)
                            ctxStat = ctxLine(used, cmax) + if (nkf >= 0) "  ${nkf}f" else ""
                            if (txt.isNotBlank()) {
                                val cm = ChatMessage("assistant", txt, "answer")
                                chatMessages = chatMessages + cm; latestMsg = cm
                            }
                        }
                    }
                } catch (e: Exception) { status = "JSON parse error" }
            }

            override fun onFailure(webSocket: WebSocket, t: Throwable, response: Response?) {
                status = "Error: ${t.localizedMessage}"; connected = false
            }
        }
    }

    // open the socket for the session; (re)send the task on open/change
    LaunchedEffect(connected) {
        if (connected) {
            webSocket = client.newWebSocket(Request.Builder().url(serverUrl).build(), wsListener)
        } else {
            recording = false
            webSocket?.close(1000, "session ended"); webSocket = null
        }
    }
    LaunchedEffect(task, connected) {
        if (connected && task.isNotBlank())
            webSocket?.send(JSONObject().put("type", "task").put("task", task).toString())
    }

    // periodic ping (every second)
    LaunchedEffect(Unit) {
        while (true) {
            val start = System.currentTimeMillis()
            client.newCall(Request.Builder().url("$pingUrl?t=$start").build()).enqueue(object : Callback {
                override fun onFailure(call: Call, e: IOException) { pingMs = "Ping: -- ms" }
                override fun onResponse(call: Call, response: Response) {
                    pingMs = "Ping: ${System.currentTimeMillis() - start} ms"; response.close()
                }
            })
            delay(1000)
        }
    }

    // data-usage stats (every 500ms)
    LaunchedEffect(Unit) {
        var lastTime = System.currentTimeMillis(); var lastSent = 0L; var lastRecv = 0L
        while (true) {
            delay(500)
            val nowMs = System.currentTimeMillis()
            val sent = totalSentBytes.get(); val recv = totalRecvBytes.get()
            val dt = (nowMs - lastTime).coerceAtLeast(1) / 1000.0
            val rate = ((sent - lastSent) + (recv - lastRecv)) / 1024.0 / dt
            dataStats = "Sent %.1fMB  Recv %.1fMB  %.0f KB/s".format(
                sent / 1048576.0, recv / 1048576.0, rate)
            lastTime = nowMs; lastSent = sent; lastRecv = recv
        }
    }

    // --- CameraX (no ARCore) ---
    val lifecycleOwner = LocalLifecycleOwner.current
    val analysisExecutor = remember { Executors.newSingleThreadExecutor() }
    DisposableEffect(Unit) { onDispose { analysisExecutor.shutdown() } }

    Box(Modifier.fillMaxSize()) {
        AndroidView(
            modifier = Modifier.fillMaxSize(),
            factory = { c ->
                val previewView = PreviewView(c).apply { scaleType = PreviewView.ScaleType.FILL_CENTER }
                val future = ProcessCameraProvider.getInstance(c)
                future.addListener({
                    val provider = future.get()
                    val preview = Preview.Builder().build()
                    preview.setSurfaceProvider(previewView.surfaceProvider)
                    val analysis = ImageAnalysis.Builder()
                        .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                        .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_YUV_420_888)
                        .build()
                    analysis.setAnalyzer(analysisExecutor) { img -> controller.analyze(img) }
                    try {
                        provider.unbindAll()
                        provider.bindToLifecycle(
                            lifecycleOwner, CameraSelector.DEFAULT_BACK_CAMERA, preview, analysis)
                    } catch (e: Exception) { Log.e("Camera", "bind failed", e) }
                }, ContextCompat.getMainExecutor(c))
                previewView
            }
        )

        // ---- TASK ENTRY (top) ----
        TaskInputBar(
            currentTask = task,
            onSet = { t ->
                task = t; connected = true
                chatMessages = chatMessages + ChatMessage("user", t); status = "Task set"
            },
            modifier = Modifier.align(Alignment.TopCenter).fillMaxWidth()
                .padding(top = 40.dp, start = 10.dp, end = 10.dp)
        )

        // ---- STATS (top-right) ----
        Column(
            Modifier.align(Alignment.TopEnd).padding(top = 148.dp, end = 12.dp)
                .background(Color.Black.copy(alpha = 0.7f), RoundedCornerShape(8.dp))
                .padding(horizontal = 10.dp, vertical = 6.dp),
            horizontalAlignment = Alignment.End
        ) {
            Text(roundtripLatency, color = Color.Cyan, fontSize = 12.sp)
            Text(pingMs, color = Color.Cyan, fontSize = 11.sp)
            Text(llmStats, color = Color.Cyan, fontSize = 11.sp)
            if (ctxStat.isNotBlank()) Text(ctxStat, color = Color.Cyan, fontSize = 11.sp)
            Text(dataStats, color = Color.Cyan, fontSize = 10.sp)
        }

        // ---- bottom controls (stacked; nothing overlaps) ----
        Column(
            Modifier.align(Alignment.BottomCenter).fillMaxWidth()
                .padding(bottom = 20.dp, start = 12.dp, end = 12.dp),
            horizontalAlignment = Alignment.CenterHorizontally,
            verticalArrangement = Arrangement.spacedBy(10.dp)
        ) {
            TransientMessage(
                latest = latestMsg, onExpand = { showChat = true },
                modifier = Modifier.fillMaxWidth()
            )
            Row(
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.spacedBy(20.dp)
            ) {
                SnapRecordButton(
                    recording = recording,
                    onSnap = {
                        if (connected) {
                            webSocket?.send(JSONObject().put("type", "snap").toString())
                            controller.snapRequested = true
                        } else status = "Set a task first"
                    },
                    onRecordStart = { recording = true },
                    onRecordStop = { recording = false }
                )
                Button(onClick = { showChat = !showChat }) { Text(if (showChat) "Hide" else "Chat") }
            }
            Row(
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.spacedBy(6.dp),
                modifier = Modifier.background(Color.Black.copy(alpha = 0.5f), RoundedCornerShape(8.dp)).padding(4.dp)
            ) {
                listOf(1 to "1", 2 to "1/2", 4 to "1/4").forEach { (factor, label) ->
                    Button(
                        onClick = { scaleFactor = factor },
                        colors = ButtonDefaults.buttonColors(
                            containerColor = if (scaleFactor == factor) Color.Cyan else Color.Transparent,
                            contentColor = if (scaleFactor == factor) Color.Black else Color.White
                        ),
                        contentPadding = PaddingValues(horizontal = 10.dp, vertical = 2.dp)
                    ) { Text(label, fontSize = 12.sp) }
                }
                Button(
                    onClick = {
                        webSocket?.send(JSONObject().put("type", "reset").toString())
                        chatMessages = emptyList(); latestMsg = null; status = "Session reset"
                    },
                    contentPadding = PaddingValues(horizontal = 10.dp, vertical = 2.dp)
                ) { Text("Reset", fontSize = 12.sp) }
            }
            Text("$status  ·  $pingMs", color = Color.Yellow, fontSize = 11.sp)
        }

        // ---- full chat sheet (overlay) with a question box ----
        if (showChat) ChatSheet(
            messages = chatMessages,
            onClose = { showChat = false },
            onAsk = { q ->
                if (connected) {
                    webSocket?.send(JSONObject().put("type", "ask").put("q", q).toString())
                    chatMessages = chatMessages + ChatMessage("user", q)
                } else status = "Set a task first"
            },
            modifier = Modifier.align(Alignment.BottomCenter)
        )
    }
}