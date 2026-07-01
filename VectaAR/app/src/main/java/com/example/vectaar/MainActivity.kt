package com.example.vectaar

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.ImageFormat
import android.graphics.Rect
import android.graphics.YuvImage
import android.media.Image
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
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import okhttp3.*
import okio.ByteString.Companion.toByteString
import org.json.JSONObject
import java.io.ByteArrayOutputStream
import java.io.IOException
import java.nio.ByteBuffer
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicLong
import kotlin.time.Duration.Companion.seconds
import io.github.sceneview.rememberEngine
import io.github.sceneview.rememberModelLoader
import io.github.sceneview.rememberModelInstance

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
            else Box(modifier = Modifier.fillMaxSize(), contentAlignment = Alignment.Center) { Text("Camera permission required.") }
        }
    }
}

@Composable
fun ARScreen() {
    var trackingStatus by remember { mutableStateOf("Initializing AR...") }
    var isStreaming by remember { mutableStateOf(false) }
    var serverLogs by remember { mutableStateOf("Disconnected") }
    var roundtripLatency by remember { mutableStateOf("Latency: -- ms") }
    var scaleFactor by remember { mutableIntStateOf(1) }

    val targetCoord = remember { mutableStateOf<Pair<Double, Double>?>(null) }
    var targetName by remember { mutableStateOf("") }

    var hitAnchor by remember { mutableStateOf<com.google.ar.core.Anchor?>(null) }

    // --- DEBUG INDICATOR STATE ---
    var debugDotPos by remember { mutableStateOf<Offset?>(null) }
    var debugDotExpiry by remember { mutableStateOf(0L) }

    // --- TAP-TO-PLACE STATE (debug) ---
    var pendingTap by remember { mutableStateOf<Offset?>(null) }

    // --- FRAME-ENCODE OFFLOAD ---
    val scope = rememberCoroutineScope()
    val encoding = remember { AtomicBoolean(false) }

    // --- DATA USAGE TRACKING ---
    // Atomic counters are bumped from the send coroutine / the OkHttp receive thread.
    // The render loop reads them ~2x/sec and computes a throughput rate (main thread).
    val totalSentBytes = remember { AtomicLong(0L) }
    val totalRecvBytes = remember { AtomicLong(0L) }
    var dataStats by remember { mutableStateOf("Sent 0.0MB  Recv 0.0MB  0 KB/s") }
    var lastStatsTime by remember { mutableStateOf(0L) }
    var lastSentSnapshot by remember { mutableStateOf(0L) }
    var lastRecvSnapshot by remember { mutableStateOf(0L) }

    val displayMetrics = LocalContext.current.resources.displayMetrics
    val screenWidth = displayMetrics.widthPixels
    val screenHeight = displayMetrics.heightPixels

    val ip = BuildConfig.SERVER_IP
    val serverUrl = "ws://$ip/ws"
    val pingUrl = "ws://$ip/ping"

    val client = remember { OkHttpClient.Builder().readTimeout(3.seconds).build() }

    var webSocket by remember { mutableStateOf<WebSocket?>(null) }

    var lastFrameTime by remember { mutableStateOf(0L) }

    // Send interval scales with 1/scaleFactor^2 so (frames x image area) — i.e. the data
    // rate — stays roughly CONSTANT across quality settings, instead of full-res frames
    // backpressuring the pipeline. Anchored so 1/4 (scaleFactor=4) -> 100ms.
    //   scaleFactor 1 (full)  -> 1600ms
    //   scaleFactor 2 (1/2)   -> 400ms
    //   scaleFactor 4 (1/4)   -> 100ms
    val baseIntervalMs = 1600L

    val engine = rememberEngine()
    val modelLoader = rememberModelLoader(engine = engine)

    val boxModel = rememberModelInstance(
        modelLoader = modelLoader,
        fileLocation = "https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Models/master/2.0/Box/glTF-Binary/Box.glb"
    )

    // --- Pulsing animation for the debug indicator ---
    val pulse = rememberInfiniteTransition(label = "dot")
    val pulseRadius by pulse.animateFloat(
        initialValue = 20f,
        targetValue = 50f,
        animationSpec = infiniteRepeatable(tween(600), RepeatMode.Reverse),
        label = "r"
    )
    val pulseAlpha by pulse.animateFloat(
        initialValue = 1f,
        targetValue = 0.2f,
        animationSpec = infiniteRepeatable(tween(600), RepeatMode.Reverse),
        label = "a"
    )

    val wsListener = remember {
        object : WebSocketListener() {
            override fun onOpen(webSocket: WebSocket, response: Response) {
                serverLogs = "Connected to Vision Engine"
            }

            override fun onMessage(webSocket: WebSocket, text: String) {
                // Count received bytes (chars ~= bytes for ASCII JSON; good enough).
                totalRecvBytes.addAndGet(text.length.toLong())
                try {
                    val json = JSONObject(text)
                    if (json.has("timestamp")) {
                        val rtt = System.currentTimeMillis() - json.getLong("timestamp")
                        roundtripLatency = "Latency: ${rtt} ms"
                    }
                    if (json.optBoolean("target_found", false)) {
                        val xNorm = json.getDouble("x_norm")
                        val yNorm = json.getDouble("y_norm")
                        targetName = json.getString("item_name")
                        targetCoord.value = Pair(xNorm, yNorm)
                    } else {
                        serverLogs = "Searching for objects..."
                    }
                } catch (e: Exception) {
                    serverLogs = "JSON Parse Error"
                }
            }

            override fun onFailure(webSocket: WebSocket, t: Throwable, response: Response?) {
                serverLogs = "Error: ${t.localizedMessage}"
                isStreaming = false
            }
        }
    }

    LaunchedEffect(isStreaming) {
        if (isStreaming) {
            hitAnchor = null
            targetCoord.value = null
            val request = Request.Builder().url(serverUrl).build()
            webSocket = client.newWebSocket(request, wsListener)
        } else {
            webSocket?.close(1000, "User stopped stream")
            webSocket = null
            serverLogs = if (targetName.isNotEmpty()) "Found: ${targetName.uppercase()}!" else "Stream Stopped"
        }
    }

    Box(modifier = Modifier.fillMaxSize()) {
        ARSceneView(
            modifier = Modifier.fillMaxSize(),
            // CRITICAL: share OUR engine/modelLoader with the scene (fixes invisible box).
            engine = engine,
            modelLoader = modelLoader,
            planeRenderer = true,
            onSessionUpdated = { session, frame ->
                val camera = frame.camera
                trackingStatus = if (camera.trackingState == TrackingState.TRACKING) "Tracking: Good" else "Tracking: Paused"

                // --- DATA USAGE STATS (recompute ~2x/sec from atomic counters) ---
                run {
                    val nowMs = System.currentTimeMillis()
                    if (lastStatsTime == 0L) lastStatsTime = nowMs
                    if (nowMs - lastStatsTime >= 500) {
                        val sent = totalSentBytes.get()
                        val recv = totalRecvBytes.get()
                        val dtSec = (nowMs - lastStatsTime) / 1000.0
                        val rateKbps = ((sent - lastSentSnapshot) + (recv - lastRecvSnapshot)) / 1024.0 / dtSec
                        dataStats = "Sent %.1fMB  Recv %.1fMB  %.0f KB/s".format(
                            sent / 1048576.0, recv / 1048576.0, rateKbps
                        )
                        lastStatsTime = nowMs
                        lastSentSnapshot = sent
                        lastRecvSnapshot = recv
                    }
                }

                // --- TAP-TO-PLACE (debug) ---
                val tap = pendingTap
                if (tap != null && camera.trackingState == TrackingState.TRACKING) {
                    pendingTap = null
                    val tapHits = frame.hitTest(tap.x, tap.y)
                    val planeHit = tapHits.firstOrNull {
                        it.trackable is Plane && (it.trackable as Plane).isPoseInPolygon(it.hitPose)
                    }
                    if (planeHit != null) {
                        hitAnchor = planeHit.createAnchor()
                    } else {
                        val pose = camera.pose.compose(Pose.makeTranslation(0f, 0f, -1f))
                        hitAnchor = session.createAnchor(pose)
                    }
                    debugDotPos = tap
                    debugDotExpiry = System.currentTimeMillis() + 2000L
                }

                // --- PLACEMENT LOGIC (from detection) ---
                val currentTarget = targetCoord.value
                if (currentTarget != null && camera.trackingState == TrackingState.TRACKING && hitAnchor == null) {
                    val inputCoords = floatArrayOf(
                        currentTarget.first.toFloat(),
                        currentTarget.second.toFloat()
                    )
                    val outputCoords = FloatArray(2)
                    frame.transformCoordinates2d(
                        com.google.ar.core.Coordinates2d.IMAGE_NORMALIZED,
                        inputCoords,
                        com.google.ar.core.Coordinates2d.VIEW,
                        outputCoords
                    )

                    val hitX = outputCoords[0]
                    val hitY = outputCoords[1]

                    debugDotPos = Offset(hitX, hitY)
                    debugDotExpiry = System.currentTimeMillis() + 2000L

                    // --- REAL HIT TEST ---
                    val hitResults = frame.hitTest(hitX, hitY)
                    val firstValidHit = hitResults.firstOrNull { hit ->
                        hit.trackable is Plane && (hit.trackable as Plane).isPoseInPolygon(hit.hitPose)
                    }
                    if (firstValidHit != null) {
                        hitAnchor = firstValidHit.createAnchor()
                        Log.d("ARHit", "PLACED on plane at view($hitX, $hitY) | boxModel=${boxModel != null}")
                        targetCoord.value = null
                        isStreaming = false
                    } else {
                        Log.d("ARHit", "Detection at view($hitX, $hitY) but no plane there (hits=${hitResults.size}); waiting for next")
                        targetCoord.value = null
                    }
                }

                // Expire the debug dot.
                if (debugDotPos != null && System.currentTimeMillis() > debugDotExpiry) {
                    debugDotPos = null
                }

                // --- STREAMING LOOP (encode offloaded; data-rate-constant interval) ---
                if (isStreaming && camera.trackingState == TrackingState.TRACKING) {
                    val currentTime = System.currentTimeMillis()
                    val frameIntervalMs = baseIntervalMs / (scaleFactor.toLong() * scaleFactor.toLong())
                    if (currentTime - lastFrameTime > frameIntervalMs && encoding.compareAndSet(false, true)) {
                        lastFrameTime = currentTime
                        var launched = false
                        try {
                            frame.acquireCameraImage().use { image ->
                                if (image.format == ImageFormat.YUV_420_888) {
                                    val yuv = ImageUtils.extractYuv(image, currentTime)
                                    launched = true
                                    scope.launch(Dispatchers.Default) {
                                        try {
                                            val jpeg = ImageUtils.yuvToDownsampledJpeg(yuv, scaleFactor)
                                            jpeg?.let {
                                                val buffer = ByteBuffer.allocate(8 + it.size)
                                                buffer.putLong(yuv.timestamp)
                                                buffer.put(it)
                                                webSocket?.send(buffer.array().toByteString())
                                                totalSentBytes.addAndGet((8 + it.size).toLong())
                                            }
                                        } catch (e: Exception) {
                                            Log.e("ARStream", "Encode/send failed", e)
                                        } finally {
                                            encoding.set(false)
                                        }
                                    }
                                }
                            }
                        } catch (e: Exception) {
                            Log.e("ARStream", "Frame acquisition failed", e)
                        } finally {
                            if (!launched) encoding.set(false)
                        }
                    }
                }
            }
        ) {
            // --- SCENE LIGHT ---
            LightNode(
                engine = engine,
                type = LightManager.Type.DIRECTIONAL,
                apply = {
                    color(1.0f, 1.0f, 1.0f)
                    intensity(100_000f)
                    direction(0.0f, -1.0f, -1.0f)
                    castShadows(false)
                }
            )

            // --- DECLARATIVE 3D NODES ---
            hitAnchor?.let { anchor ->
                AnchorNode(anchor = anchor) {
                    boxModel?.let { modelInstance ->
                        ModelNode(
                            modelInstance = modelInstance,
                            scaleToUnits = 0.1f
                        )
                    }
                }
            }
        }

        // --- TAP CATCHER (debug) ---
        Box(
            modifier = Modifier
                .fillMaxSize()
                .pointerInput(Unit) {
                    detectTapGestures { offset -> pendingTap = offset }
                }
        )

        // --- DEBUG INDICATOR OVERLAY (2D, pure Compose) ---
        debugDotPos?.let { pos ->
            Canvas(modifier = Modifier.fillMaxSize()) {
                drawCircle(
                    color = Color.Magenta.copy(alpha = pulseAlpha),
                    radius = pulseRadius,
                    center = pos,
                    style = Stroke(width = 4f)
                )
                drawCircle(
                    color = Color.Magenta,
                    radius = 8f,
                    center = pos
                )
            }
        }

        // Latency + Data Usage Overlay
        Column(
            modifier = Modifier.align(Alignment.TopEnd).padding(top = 48.dp, end = 16.dp)
                .background(Color.Black.copy(alpha = 0.7f), RoundedCornerShape(8.dp))
                .padding(horizontal = 12.dp, vertical = 6.dp),
            horizontalAlignment = Alignment.End
        ) {
            Text(text = roundtripLatency, color = Color.Cyan, fontSize = 14.sp)
            Text(text = dataStats, color = Color.Cyan, fontSize = 11.sp)
        }

        // Control Panel
        Column(
            modifier = Modifier.align(Alignment.BottomCenter).fillMaxWidth().padding(16.dp),
            verticalArrangement = Arrangement.spacedBy(12.dp),
            horizontalAlignment = Alignment.CenterHorizontally
        ) {
            Text(text = "Tap anywhere to drop a test box", color = Color.White, fontSize = 12.sp)
            Text(text = trackingStatus, color = Color.Green)
            Text(text = serverLogs, color = if (targetName.isNotEmpty()) Color.Green else Color.Yellow, fontSize = 18.sp)
            Row(
                horizontalArrangement = Arrangement.spacedBy(8.dp),
                modifier = Modifier.background(Color.Black.copy(alpha = 0.5f), RoundedCornerShape(8.dp)).padding(4.dp)
            ) {
                // Labels show the QUALITY fraction (1 = full, 1/2, 1/4) so it's intuitive
                // that higher settings reduce frame quality.
                listOf(1, 2, 4).forEach { factor ->
                    val label = when (factor) {
                        1 -> "1"
                        2 -> "1/2"
                        4 -> "1/4"
                        else -> "1/$factor"
                    }
                    Button(
                        onClick = { scaleFactor = factor },
                        colors = ButtonDefaults.buttonColors(
                            containerColor = if (scaleFactor == factor) Color.Cyan else Color.Transparent,
                            contentColor = if (scaleFactor == factor) Color.Black else Color.White
                        ),
                        contentPadding = PaddingValues(horizontal = 12.dp, vertical = 4.dp)
                    ) {
                        Text(text = label)
                    }
                }
            }
            Row(
                horizontalArrangement = Arrangement.spacedBy(16.dp)
            ) {
                Button(onClick = {
                    serverLogs = "Pinging..."
                    try {
                        val request = Request.Builder().url(pingUrl).build()
                        client.newCall(request).enqueue(object : Callback {
                            override fun onFailure(call: Call, e: IOException) {
                                serverLogs = "Ping Failed: ${e.localizedMessage}"
                            }

                            override fun onResponse(call: Call, response: Response) {
                                serverLogs = "Ping Success: ${response.code}"
                            }
                        })
                    } catch (e: IllegalArgumentException) {
                        serverLogs = "Invalid URL Format"
                    }
                }) {
                    Text("Test Ping")
                }
                Button(onClick = { isStreaming = !isStreaming }) {
                    Text(if (isStreaming) "Stop Scanning" else "Start Scanning")
                }
            }
        }
    }
}

object ImageUtils {

    class Yuv420Frame(
        val width: Int,
        val height: Int,
        val y: ByteArray,
        val u: ByteArray,
        val v: ByteArray,
        val yRowStride: Int,
        val yPixelStride: Int,
        val uRowStride: Int,
        val uPixelStride: Int,
        val vRowStride: Int,
        val vPixelStride: Int,
        val timestamp: Long
    )

    fun extractYuv(image: Image, timestamp: Long): Yuv420Frame {
        val p = image.planes
        val yb = p[0].buffer.duplicate()
        val ub = p[1].buffer.duplicate()
        val vb = p[2].buffer.duplicate()
        val y = ByteArray(yb.remaining()); yb.get(y)
        val u = ByteArray(ub.remaining()); ub.get(u)
        val v = ByteArray(vb.remaining()); vb.get(v)
        return Yuv420Frame(
            image.width, image.height, y, u, v,
            p[0].rowStride, p[0].pixelStride,
            p[1].rowStride, p[1].pixelStride,
            p[2].rowStride, p[2].pixelStride,
            timestamp
        )
    }

    fun yuvToDownsampledJpeg(f: Yuv420Frame, scaleFactor: Int): ByteArray? {
        val newWidth = f.width / scaleFactor
        val newHeight = f.height / scaleFactor
        val ySize = newWidth * newHeight
        val nv21 = ByteArray(ySize + (ySize / 2))

        var outIdx = 0

        for (row in 0 until newHeight) {
            val srcRow = row * scaleFactor
            for (col in 0 until newWidth) {
                val srcCol = col * scaleFactor
                nv21[outIdx++] = f.y[srcRow * f.yRowStride + srcCol * f.yPixelStride]
            }
        }

        for (row in 0 until newHeight / 2) {
            val srcRow = row * scaleFactor
            for (col in 0 until newWidth / 2) {
                val srcCol = col * scaleFactor
                nv21[outIdx++] = f.v[srcRow * f.vRowStride + srcCol * f.vPixelStride]
                nv21[outIdx++] = f.u[srcRow * f.uRowStride + srcCol * f.uPixelStride]
            }
        }

        val out = ByteArrayOutputStream()
        val yuvImage = YuvImage(nv21, ImageFormat.NV21, newWidth, newHeight, null)
        yuvImage.compressToJpeg(Rect(0, 0, newWidth, newHeight), 80, out)
        return out.toByteArray()
    }
}