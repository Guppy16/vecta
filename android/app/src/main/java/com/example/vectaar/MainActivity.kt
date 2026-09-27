package com.example.vectaar

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.ImageFormat
import android.os.Bundle
import android.util.Log
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.animation.core.*
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.gestures.detectTapGestures
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.Text
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.core.content.ContextCompat
import com.google.android.filament.LightManager
import com.google.ar.core.Coordinates2d
import com.google.ar.core.Plane
import com.google.ar.core.Pose
import com.google.ar.core.TrackingState
import io.github.sceneview.ar.ARSceneView
import io.github.sceneview.ar.node.AnchorNode
import io.github.sceneview.node.LightNode
import io.github.sceneview.node.ModelNode
import io.github.sceneview.rememberEngine
import io.github.sceneview.rememberModelInstance
import io.github.sceneview.rememberModelLoader
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import okhttp3.*
import okio.ByteString.Companion.toByteString
import org.json.JSONObject
import java.io.IOException
import java.nio.ByteBuffer
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicLong
import kotlin.time.Duration.Companion.seconds

private const val BOX_GLB =
    "https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Models/master/2.0/Box/glTF-Binary/Box.glb"

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
            if (hasCameraPermission) ARScreen()
            else Box(Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
                Text("Camera permission required.")
            }
        }
    }
}

@Composable
fun ARScreen() {
    // --- task / session ---
    var task by remember { mutableStateOf("") }
    var connected by remember { mutableStateOf(false) }
    var recording by remember { mutableStateOf(false) }
    var snapRequested by remember { mutableStateOf(false) }
    var chatMessages by remember { mutableStateOf(listOf<ChatMessage>()) }
    var latestMsg by remember { mutableStateOf<ChatMessage?>(null) }
    var showChat by remember { mutableStateOf(false) }
    var expandedImage by remember { mutableStateOf<ByteArray?>(null) }
    var scaleFactor by remember { mutableIntStateOf(4) }
    var status by remember { mutableStateOf("Set a task to begin") }
    var trackingStatus by remember { mutableStateOf("Initializing AR...") }

    // --- telemetry ---
    var llmStats by remember { mutableStateOf("") }
    var pingMs by remember { mutableStateOf("Ping: -- ms") }
    var roundtripLatency by remember { mutableStateOf("RTT: -- ms") }
    var ctxStat by remember { mutableStateOf("") }
    val totalSentBytes = remember { AtomicLong(0L) }
    val totalRecvBytes = remember { AtomicLong(0L) }
    var dataStats by remember { mutableStateOf("Sent 0.0MB  Recv 0.0MB  0 KB/s") }

    // --- AR: multiple anchors (one per detected object) ---
    val targetCoord = remember { mutableStateOf<Pair<Double, Double>?>(null) }
    var anchors by remember { mutableStateOf(listOf<com.google.ar.core.Anchor>()) }
    var debugDotPos by remember { mutableStateOf<Offset?>(null) }
    var debugDotExpiry by remember { mutableStateOf(0L) }
    var pendingTap by remember { mutableStateOf<Offset?>(null) }

    // recently-sent frames (ts -> jpeg), so a response can be paired with its frame
    val sentFrames = remember { java.util.Collections.synchronizedList(ArrayList<Pair<Long, ByteArray>>()) }

    val scope = rememberCoroutineScope()
    val encoding = remember { AtomicBoolean(false) }
    var lastFrameTime by remember { mutableStateOf(0L) }

    val ip = BuildConfig.SERVER_IP
    val serverUrl = "ws://$ip/ws"
    val pingUrl = "http://$ip/ping"
    val client = remember { OkHttpClient.Builder().readTimeout(3.seconds).build() }
    var webSocket by remember { mutableStateOf<WebSocket?>(null) }

    val baseIntervalMs = 6400L        // /scaleFactor^2 -> 400ms at 1/4x

    val engine = rememberEngine()
    val modelLoader = rememberModelLoader(engine = engine)

    val pulse = rememberInfiniteTransition(label = "dot")
    val pulseRadius by pulse.animateFloat(20f, 50f,
        infiniteRepeatable(tween(600), RepeatMode.Reverse), label = "r")
    val pulseAlpha by pulse.animateFloat(1f, 0.2f,
        infiniteRepeatable(tween(600), RepeatMode.Reverse), label = "a")

    val wsListener = remember {
        object : WebSocketListener() {
            override fun onOpen(webSocket: WebSocket, response: Response) { status = "Connected" }

            override fun onMessage(webSocket: WebSocket, text: String) {
                totalRecvBytes.addAndGet(text.length.toLong())
                try {
                    val json = JSONObject(text)
                    when (json.optString("type")) {
                        "llm" -> {
                            val m = parseLlm(json)
                            llmStats = "LLM ${m.latencyMs}ms" + (m.tokPerS?.let { "  $it tok/s" } ?: "")
                            ctxStat = ctxLine(m.ctxTokens, m.ctxMax)
                            val ts = json.optLong("ts", -1L)
                            if (ts >= 0)
                                roundtripLatency = "RTT: ${System.currentTimeMillis() - ts} ms"
                            // pair the verdict with the exact frame the model judged
                            val frameBytes = synchronized(sentFrames) {
                                sentFrames.firstOrNull { it.first == ts }?.second
                            }
                            val cm = ChatMessage("model", m.say, m.status, frameBytes, m.point)
                            chatMessages = (chatMessages + cm).takeLast(120)
                            if (m.say.isNotBlank()) latestMsg = cm      // transient only for found/info
                            if (m.status == "found" && m.point != null) {
                                targetCoord.value = m.point
                                status = "Found — placing anchor"
                            }
                        }
                        "answer" -> {
                            val txt = json.optString("text", "")
                            llmStats = "LLM ${json.optInt("latency_ms", 0)}ms"
                            val used = if (json.isNull("ctx_tokens")) null else json.optInt("ctx_tokens")
                            val cmax = if (json.isNull("ctx_max")) null else json.optInt("ctx_max")
                            val nkf = json.optInt("n_keyframes", -1)
                            ctxStat = ctxLine(used, cmax) + if (nkf >= 0) "  ${nkf}f" else ""
                            if (txt.isNotBlank()) {
                                val cm = ChatMessage("answer", txt, "answer")
                                chatMessages = (chatMessages + cm).takeLast(120); latestMsg = cm
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

    LaunchedEffect(Unit) {              // periodic ping
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

    LaunchedEffect(Unit) {              // data-usage stats
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

    Box(Modifier.fillMaxSize()) {
        ARSceneView(
            modifier = Modifier.fillMaxSize(),
            engine = engine,
            modelLoader = modelLoader,
            planeRenderer = true,
            onSessionUpdated = { session, frame ->
                val camera = frame.camera
                trackingStatus =
                    if (camera.trackingState == TrackingState.TRACKING) "Tracking: Good" else "Tracking: Paused"
                val now = System.currentTimeMillis()

                // tap-to-place (debug) -> adds an anchor
                val tap = pendingTap
                if (tap != null && camera.trackingState == TrackingState.TRACKING) {
                    pendingTap = null
                    val planeHit = frame.hitTest(tap.x, tap.y).firstOrNull {
                        it.trackable is Plane && (it.trackable as Plane).isPoseInPolygon(it.hitPose)
                    }
                    val a = planeHit?.createAnchor()
                        ?: session.createAnchor(camera.pose.compose(Pose.makeTranslation(0f, 0f, -1f)))
                    anchors = anchors + a
                    debugDotPos = tap; debugDotExpiry = now + 2000L
                }

                // place from the LLM's found point -> adds an anchor
                val currentTarget = targetCoord.value
                if (currentTarget != null && camera.trackingState == TrackingState.TRACKING) {
                    targetCoord.value = null
                    val out = FloatArray(2)
                    frame.transformCoordinates2d(
                        Coordinates2d.IMAGE_NORMALIZED,
                        floatArrayOf(currentTarget.first.toFloat(), currentTarget.second.toFloat()),
                        Coordinates2d.VIEW, out)
                    debugDotPos = Offset(out[0], out[1]); debugDotExpiry = now + 3000L
                    val hits = frame.hitTest(out[0], out[1])
                    val hit = hits.firstOrNull {
                        it.trackable is Plane && (it.trackable as Plane).isPoseInPolygon(it.hitPose)
                    } ?: hits.firstOrNull()
                    if (hit != null) anchors = anchors + hit.createAnchor()
                    else Log.d("ARHit", "no plane/feature at the detected point yet")
                }

                if (debugDotPos != null && now > debugDotExpiry) debugDotPos = null

                // snap / record frame send (400ms at 1/4x)
                if (connected && camera.trackingState == TrackingState.TRACKING) {
                    val interval = baseIntervalMs / (scaleFactor.toLong() * scaleFactor.toLong())
                    val doRecord = recording && (now - lastFrameTime >= interval)
                    if ((doRecord || snapRequested) && encoding.compareAndSet(false, true)) {
                        if (doRecord) lastFrameTime = now
                        snapRequested = false
                        var launched = false
                        try {
                            frame.acquireCameraImage().use { image ->
                                if (image.format == ImageFormat.YUV_420_888) {
                                    val yuv = ImageUtils.extractYuv(image, now)
                                    launched = true
                                    scope.launch(Dispatchers.Default) {
                                        try {
                                            ImageUtils.yuvToDownsampledJpeg(yuv, scaleFactor)?.let { jpeg ->
                                                val b = ByteBuffer.allocate(8 + jpeg.size)
                                                b.putLong(yuv.timestamp); b.put(jpeg)
                                                webSocket?.send(b.array().toByteString())
                                                totalSentBytes.addAndGet((8 + jpeg.size).toLong())
                                                synchronized(sentFrames) {   // keep for the chat log
                                                    sentFrames.add(yuv.timestamp to jpeg)
                                                    if (sentFrames.size > 60) sentFrames.removeAt(0)
                                                }
                                            }
                                        } catch (e: Exception) { Log.e("LLM", "send failed", e) }
                                        finally { encoding.set(false) }
                                    }
                                }
                            }
                        } catch (e: Exception) {
                            Log.e("LLM", "acquire failed", e)
                        } finally { if (!launched) encoding.set(false) }
                    }
                }
            }
        ) {
            LightNode(
                engine = engine, type = LightManager.Type.DIRECTIONAL,
                apply = {
                    color(1.0f, 1.0f, 1.0f); intensity(100_000f)
                    direction(0.0f, -1.0f, -1.0f); castShadows(false)
                }
            )
            // one box per anchor; each gets its OWN model instance (keyed) so they coexist
            anchors.forEach { anchor ->
                key(anchor) {
                    val inst = rememberModelInstance(modelLoader = modelLoader, fileLocation = BOX_GLB)
                    AnchorNode(anchor = anchor) {
                        inst?.let { ModelNode(modelInstance = it, scaleToUnits = 0.1f) }
                    }
                }
            }
        }

        // tap catcher (debug)
        Box(Modifier.fillMaxSize().pointerInput(Unit) {
            detectTapGestures { offset -> pendingTap = offset }
        })

        // detection-point / placement indicator
        debugDotPos?.let { pos ->
            Canvas(Modifier.fillMaxSize()) {
                drawCircle(Color.Magenta.copy(alpha = pulseAlpha), pulseRadius, pos, style = Stroke(4f))
                drawCircle(Color.Magenta, 8f, pos)
            }
        }

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

        // ---- bottom controls ----
        Column(
            Modifier.align(Alignment.BottomCenter).fillMaxWidth()
                .padding(bottom = 20.dp, start = 12.dp, end = 12.dp),
            horizontalAlignment = Alignment.CenterHorizontally,
            verticalArrangement = Arrangement.spacedBy(10.dp)
        ) {
            TransientMessage(latest = latestMsg, onExpand = { showChat = true },
                modifier = Modifier.fillMaxWidth())
            Row(
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.spacedBy(20.dp)
            ) {
                SnapRecordButton(
                    recording = recording,
                    onSnap = {
                        if (connected) {
                            webSocket?.send(JSONObject().put("type", "snap").toString())
                            snapRequested = true
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
                        chatMessages = emptyList(); latestMsg = null
                        anchors.forEach { it.detach() }; anchors = emptyList()
                        synchronized(sentFrames) { sentFrames.clear() }
                        status = "Session reset"
                    },
                    contentPadding = PaddingValues(horizontal = 10.dp, vertical = 2.dp)
                ) { Text("Reset", fontSize = 12.sp) }
            }
            Text("$status  ·  $trackingStatus", color = Color.Yellow, fontSize = 11.sp)
        }

        // ---- chat sheet + full-screen frame viewer ----
        if (showChat) ChatSheet(
            messages = chatMessages,
            onClose = { showChat = false },
            onAsk = { q ->
                if (connected) {
                    webSocket?.send(JSONObject().put("type", "ask").put("q", q).toString())
                    chatMessages = chatMessages + ChatMessage("user", q)
                } else status = "Set a task first"
            },
            onImageTap = { expandedImage = it },
            modifier = Modifier.align(Alignment.BottomCenter)
        )
        expandedImage?.let { bytes ->
            ExpandedImage(bytes = bytes, onDismiss = { expandedImage = null })
        }
    }
}