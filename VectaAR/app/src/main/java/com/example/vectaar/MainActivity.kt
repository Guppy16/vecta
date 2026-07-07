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
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.core.content.ContextCompat
import com.google.android.filament.LightManager
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
import kotlinx.coroutines.launch
import okhttp3.*
import okio.ByteString.Companion.toByteString
import org.json.JSONObject
import java.io.IOException
import java.nio.ByteBuffer
import java.util.concurrent.atomic.AtomicBoolean
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
            if (hasCameraPermission) ARScreen()
            else Box(Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
                Text("Camera permission required.")
            }
        }
    }
}

@Composable
fun ARScreen() {
    // ----------------------------------------------------------------- //
    //  MAIN PATH: task -> snap/record frames -> vision LLM -> chat        //
    // ----------------------------------------------------------------- //
    var task by remember { mutableStateOf("") }
    var connected by remember { mutableStateOf(false) }          // socket/session open
    var recording by remember { mutableStateOf(false) }          // hold-to-record active
    var snapRequested by remember { mutableStateOf(false) }      // one-shot capture pending
    var recordLastFrame by remember { mutableStateOf(0L) }
    var chatMessages by remember { mutableStateOf(listOf<ChatMessage>()) }
    var latestMsg by remember { mutableStateOf<ChatMessage?>(null) }
    var showChat by remember { mutableStateOf(false) }
    var llmStats by remember { mutableStateOf("") }
    var pingMs by remember { mutableStateOf("Ping: -- ms") }
    var roundtripLatency by remember { mutableStateOf("RTT: -- ms") }
    var status by remember { mutableStateOf("Set a task to begin") }
    var trackingStatus by remember { mutableStateOf("Initializing AR...") }
    var scaleFactor by remember { mutableIntStateOf(4) }          // default 1/4x

    // --- data usage counters (bumped off-thread, read ~2x/sec on render) ---
    val totalSentBytes = remember { AtomicLong(0L) }
    val totalRecvBytes = remember { AtomicLong(0L) }
    var dataStats by remember { mutableStateOf("Sent 0.0MB  Recv 0.0MB  0 KB/s") }
    var lastStatsTime by remember { mutableStateOf(0L) }
    var lastSentSnapshot by remember { mutableStateOf(0L) }
    var lastRecvSnapshot by remember { mutableStateOf(0L) }

    // --- frame-encode offload guard (one in flight) ---
    val scope = rememberCoroutineScope()
    val encoding = remember { AtomicBoolean(false) }

    // ----------------------------------------------------------------- //
    //  DORMANT: kept but not on the main path                            //
    //   - AR placement from a detection message (old YOLO-World path)    //
    //   - hand tracking lives in HandOverlay.kt and is not wired here    //
    // ----------------------------------------------------------------- //
    val targetCoord = remember { mutableStateOf<Pair<Double, Double>?>(null) }
    var targetName by remember { mutableStateOf("") }
    var hitAnchor by remember { mutableStateOf<com.google.ar.core.Anchor?>(null) }
    var debugDotPos by remember { mutableStateOf<Offset?>(null) }
    var debugDotExpiry by remember { mutableStateOf(0L) }
    var pendingTap by remember { mutableStateOf<Offset?>(null) }

    val ip = BuildConfig.SERVER_IP
    val serverUrl = "ws://$ip/ws"
    val pingUrl = "http://$ip/ping"     // GET endpoint (was ws:// — that was a bug)
    val client = remember { OkHttpClient.Builder().readTimeout(3.seconds).build() }
    var webSocket by remember { mutableStateOf<WebSocket?>(null) }

    // Send interval scales as 1/scaleFactor^2 so (frames x area) — the data rate —
    // stays constant across quality settings. Anchored so 1/4x -> 400ms.
    //   1x -> 6400ms ; 1/2x -> 1600ms ; 1/4x -> 400ms
    val baseIntervalMs = 6400L

    val engine = rememberEngine()
    val modelLoader = rememberModelLoader(engine = engine)
    val boxModel = rememberModelInstance(
        modelLoader = modelLoader,
        fileLocation = "https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Models/master/2.0/Box/glTF-Binary/Box.glb"
    )

    // pulsing debug indicator
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

                    // --- MAIN: vision-LLM result ---
                    if (json.optString("type") == "llm") {
                        val m = parseLlm(json)
                        llmStats = "LLM ${m.latencyMs}ms" + (m.tokPerS?.let { "  $it tok/s" } ?: "")
                        if (json.has("ts"))
                            roundtripLatency = "RTT: ${System.currentTimeMillis() - json.getLong("ts")} ms"
                        if (m.say.isNotBlank()) {
                            val cm = ChatMessage("assistant", m.say, m.status)
                            chatMessages = chatMessages + cm
                            latestMsg = cm
                        }
                        return
                    }

                    // --- DORMANT: AR placement from a detection message ---
                    if (json.optBoolean("target_found", false)) {
                        targetName = json.getString("item_name")
                        targetCoord.value = Pair(json.getDouble("x_norm"), json.getDouble("y_norm"))
                    }
                } catch (e: Exception) { status = "JSON parse error" }
            }

            override fun onFailure(webSocket: WebSocket, t: Throwable, response: Response?) {
                status = "Error: ${t.localizedMessage}"; connected = false
            }
        }
    }

    // Open the socket while a session is active; (re)send the task on open/change.
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

                // --- data usage stats (~2x/sec) ---
                run {
                    val nowMs = System.currentTimeMillis()
                    if (lastStatsTime == 0L) lastStatsTime = nowMs
                    if (nowMs - lastStatsTime >= 500) {
                        val sent = totalSentBytes.get(); val recv = totalRecvBytes.get()
                        val dtSec = (nowMs - lastStatsTime) / 1000.0
                        val rate = ((sent - lastSentSnapshot) + (recv - lastRecvSnapshot)) / 1024.0 / dtSec
                        dataStats = "Sent %.1fMB  Recv %.1fMB  %.0f KB/s".format(
                            sent / 1048576.0, recv / 1048576.0, rate)
                        lastStatsTime = nowMs; lastSentSnapshot = sent; lastRecvSnapshot = recv
                    }
                }

                // --- DORMANT: tap-to-place debug box ---
                val tap = pendingTap
                if (tap != null && camera.trackingState == TrackingState.TRACKING) {
                    pendingTap = null
                    val planeHit = frame.hitTest(tap.x, tap.y).firstOrNull {
                        it.trackable is Plane && (it.trackable as Plane).isPoseInPolygon(it.hitPose)
                    }
                    hitAnchor = planeHit?.createAnchor()
                        ?: session.createAnchor(camera.pose.compose(Pose.makeTranslation(0f, 0f, -1f)))
                    debugDotPos = tap; debugDotExpiry = System.currentTimeMillis() + 2000L
                }

                // --- DORMANT: place from a detection message ---
                val currentTarget = targetCoord.value
                if (currentTarget != null && camera.trackingState == TrackingState.TRACKING && hitAnchor == null) {
                    val out = FloatArray(2)
                    frame.transformCoordinates2d(
                        com.google.ar.core.Coordinates2d.IMAGE_NORMALIZED,
                        floatArrayOf(currentTarget.first.toFloat(), currentTarget.second.toFloat()),
                        com.google.ar.core.Coordinates2d.VIEW, out)
                    debugDotPos = Offset(out[0], out[1]); debugDotExpiry = System.currentTimeMillis() + 2000L
                    frame.hitTest(out[0], out[1]).firstOrNull {
                        it.trackable is Plane && (it.trackable as Plane).isPoseInPolygon(it.hitPose)
                    }?.let { hitAnchor = it.createAnchor() }
                    targetCoord.value = null
                }

                if (debugDotPos != null && System.currentTimeMillis() > debugDotExpiry) debugDotPos = null

                // --- MAIN: snap / record frame send (record @ frameInterval; 400ms at 1/4x) ---
                if (connected && camera.trackingState == TrackingState.TRACKING) {
                    val now = System.currentTimeMillis()
                    val frameIntervalMs = baseIntervalMs / (scaleFactor.toLong() * scaleFactor.toLong())
                    val doRecord = recording && (now - recordLastFrame >= frameIntervalMs)
                    if ((doRecord || snapRequested) && encoding.compareAndSet(false, true)) {
                        if (doRecord) recordLastFrame = now
                        snapRequested = false
                        var launched = false
                        try {
                            frame.acquireCameraImage().use { image ->
                                if (image.format == ImageFormat.YUV_420_888) {
                                    val yuv = ImageUtils.extractYuv(image, now)
                                    launched = true
                                    scope.launch(Dispatchers.Default) {
                                        try {
                                            ImageUtils.yuvToDownsampledJpeg(yuv, scaleFactor)?.let {
                                                val b = ByteBuffer.allocate(8 + it.size)
                                                b.putLong(yuv.timestamp); b.put(it)
                                                webSocket?.send(b.array().toByteString())
                                                totalSentBytes.addAndGet((8 + it.size).toLong())
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
            // DORMANT: placed AR box
            hitAnchor?.let { anchor ->
                AnchorNode(anchor = anchor) {
                    boxModel?.let { ModelNode(modelInstance = it, scaleToUnits = 0.1f) }
                }
            }
        }

        // DORMANT: tap catcher for the debug box
        Box(Modifier.fillMaxSize().pointerInput(Unit) {
            detectTapGestures { offset -> pendingTap = offset }
        })

        // DORMANT: debug indicator
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
                chatMessages = chatMessages + ChatMessage("user", t)
                status = "Task set"
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
            Text(dataStats, color = Color.Cyan, fontSize = 10.sp)
        }

        // ---- bottom controls: one stacked column, nothing overlaps ----
        Column(
            Modifier.align(Alignment.BottomCenter).fillMaxWidth()
                .padding(bottom = 20.dp, start = 12.dp, end = 12.dp),
            horizontalAlignment = Alignment.CenterHorizontally,
            verticalArrangement = Arrangement.spacedBy(10.dp)
        ) {
            // latest assistant message — collapses when hidden, sits above the buttons
            TransientMessage(
                latest = latestMsg, onExpand = { showChat = true },
                modifier = Modifier.fillMaxWidth()
            )

            // primary: tap = snap, hold = record; + chat toggle
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

            // secondary: resolution + ping
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
                        val start = System.currentTimeMillis()
                        client.newCall(Request.Builder().url("$pingUrl?t=$start").build())
                            .enqueue(object : Callback {
                                override fun onFailure(call: Call, e: IOException) { pingMs = "Ping failed" }
                                override fun onResponse(call: Call, response: Response) {
                                    pingMs = "Ping: ${System.currentTimeMillis() - start} ms"; response.close()
                                }
                            })
                    },
                    contentPadding = PaddingValues(horizontal = 10.dp, vertical = 2.dp)
                ) { Text("Ping", fontSize = 12.sp) }
            }

            // status line
            Text("$status  ·  $trackingStatus", color = Color.Yellow, fontSize = 11.sp)
        }

        // ---- full chat sheet (overlay) ----
        if (showChat) ChatSheet(chatMessages, { showChat = false }, Modifier.align(Alignment.BottomCenter))
    }
}