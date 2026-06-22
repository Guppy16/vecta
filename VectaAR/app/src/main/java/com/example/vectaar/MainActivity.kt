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
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Button
import androidx.compose.material3.Text
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.core.content.ContextCompat
import com.google.ar.core.TrackingState
import io.github.sceneview.ar.ARSceneView
import okhttp3.*
import okio.ByteString.Companion.toByteString
import org.json.JSONObject
import java.io.ByteArrayOutputStream
import java.io.IOException
import java.nio.ByteBuffer
import kotlin.time.Duration.Companion.seconds
import androidx.compose.material3.ButtonDefaults // Added for visual toggle state

class MainActivity : ComponentActivity() {

    private val requestPermissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { isGranted: Boolean ->
        if (isGranted) recreate()
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        val hasCameraPermission = ContextCompat.checkSelfPermission(
            this, Manifest.permission.CAMERA
        ) == PackageManager.PERMISSION_GRANTED

        if (!hasCameraPermission) {
            requestPermissionLauncher.launch(Manifest.permission.CAMERA)
        }

        setContent {
            if (hasCameraPermission) {
                ARScreen()
            } else {
                Box(modifier = Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
                    Text("Camera permission required for AR.")
                }
            }
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

    val ip = BuildConfig.SERVER_IP
    val serverUrl = "ws://$ip/ws"
    val pingUrl = "http://$ip/ping"

    val client = remember {
        OkHttpClient.Builder()
            .readTimeout(3.seconds)
            .build()
    }

    var webSocket: WebSocket? by remember { mutableStateOf(null) }
    var lastFrameTime by remember { mutableLongStateOf(0L) }
    val frameIntervalMs = 500L

    val wsListener = remember {
        object : WebSocketListener() {
            override fun onOpen(webSocket: WebSocket, response: Response) {
                serverLogs = "Connected to WebSocket"
            }

            override fun onMessage(webSocket: WebSocket, text: String) {
                try {
                    val json = JSONObject(text)
                    if (json.has("timestamp")) {
                        val sentTime = json.getLong("timestamp")
                        val rtt = System.currentTimeMillis() - sentTime
                        roundtripLatency = "Latency: ${rtt} ms"
                    }
                    serverLogs = "Received: $text"
                } catch (e: Exception) {
                    serverLogs = "JSON Parse Error"
                }
            }

            override fun onClosing(webSocket: WebSocket, code: Int, reason: String) {
                serverLogs = "Closing: $reason"
            }

            override fun onFailure(webSocket: WebSocket, t: Throwable, response: Response?) {
                serverLogs = "Error: ${t.localizedMessage}"
                isStreaming = false
                roundtripLatency = "Latency: -- ms"
            }
        }
    }

    LaunchedEffect(isStreaming) {
        if (isStreaming) {
            val request = Request.Builder().url(serverUrl).build()
            webSocket = client.newWebSocket(request, wsListener)
        } else {
            webSocket?.close(1000, "User stopped stream")
            webSocket = null
            serverLogs = "Stream Stopped"
            roundtripLatency = "Latency: -- ms"
        }
    }

    Box(modifier = Modifier.fillMaxSize()) {
        ARSceneView(
            modifier = Modifier.fillMaxSize(),
            planeRenderer = true,
            onSessionUpdated = { _, frame ->
                val camera = frame.camera
                trackingStatus = when (camera.trackingState) {
                    TrackingState.TRACKING -> "Tracking: Good"
                    TrackingState.PAUSED -> "Tracking Paused: ${camera.trackingFailureReason.name}"
                    TrackingState.STOPPED -> "Tracking Stopped"
                    else -> "Unknown"
                }

                if (isStreaming && camera.trackingState == TrackingState.TRACKING) {
                    val currentTime = System.currentTimeMillis()
                    if (currentTime - lastFrameTime > frameIntervalMs) {
                        lastFrameTime = currentTime

                        try {
                            frame.acquireCameraImage().use { image ->
                                // Downsample factor of 4 (1920x1080 -> 480x270)
                                val smallJpegBytes = ImageUtils.yuv420ToDownsampledJpeg(image, scaleFactor)

                                smallJpegBytes?.let { jpeg ->
                                    // Allocate space for 8 bytes (Long timestamp) + image bytes
                                    val buffer = ByteBuffer.allocate(8 + jpeg.size)
                                    buffer.putLong(currentTime)
                                    buffer.put(jpeg)

                                    webSocket?.send(buffer.array().toByteString())
                                }
                            }
                        } catch (e: Exception) {
                            Log.e("ARStream", "Frame acquisition failed", e)
                        }
                    }
                }
            }
        )

        // Latency Overlay Box (Top Right Corner)
        Box(
            modifier = Modifier
                .align(Alignment.TopEnd)
                .padding(top = 48.dp, end = 16.dp)
                .background(Color.Black.copy(alpha = 0.7f), shape = RoundedCornerShape(8.dp))
                .padding(horizontal = 12.dp, vertical = 6.dp)
        ) {
            Text(
                text = roundtripLatency,
                color = if (roundtripLatency.contains("--")) Color.White else Color.Cyan,
                fontSize = 14.sp
            )
        }

        // Control Panel UI (Bottom Center)
        Column(
            modifier = Modifier
                .align(Alignment.BottomCenter)
                .fillMaxWidth()
                .padding(16.dp),
            verticalArrangement = Arrangement.spacedBy(8.dp),
            horizontalAlignment = Alignment.CenterHorizontally
        ) {
            Text(text = trackingStatus, color = Color.Green)
            Text(text = serverLogs, color = Color.Yellow)

            Row(
                horizontalArrangement = Arrangement.spacedBy(8.dp),
                modifier = Modifier.background(Color.Black.copy(alpha = 0.5f), RoundedCornerShape(8.dp)).padding(4.dp)
            ) {
                listOf(1, 2, 4).forEach { factor ->
                    Button(
                        onClick = { scaleFactor = factor },
                        colors = ButtonDefaults.buttonColors(
                            containerColor = if (scaleFactor == factor) Color.Cyan else Color.Transparent,
                            contentColor = if (scaleFactor == factor) Color.Black else Color.White
                        ),
                        contentPadding = PaddingValues(horizontal = 12.dp, vertical = 4.dp)
                    ) {
                        Text(text = "${factor}X")
                    }
                }
            }

            Row(
                horizontalArrangement = Arrangement.spacedBy(16.dp)
            ) {
                Button(onClick = {
                    serverLogs = "Pinging..."
                    val request = Request.Builder().url(pingUrl).build()
                    client.newCall(request).enqueue(object : Callback {
                        override fun onFailure(call: Call, e: IOException) {
                            serverLogs = "Ping Failed: ${e.localizedMessage}"
                        }
                        override fun onResponse(call: Call, response: Response) {
                            serverLogs = "Ping Success: ${response.code}"
                        }
                    })
                }) {
                    Text("Test Ping")
                }

                Button(onClick = { isStreaming = !isStreaming }) {
                    Text(if (isStreaming) "Stop Streaming" else "Start Streaming")
                }
            }
        }
    }
}

object ImageUtils {
    fun yuv420ToDownsampledJpeg(image: Image, scaleFactor: Int): ByteArray? {
        if (image.format != ImageFormat.YUV_420_888) return null

        val planes = image.planes
        val yBuffer = planes[0].buffer
        val uBuffer = planes[1].buffer
        val vBuffer = planes[2].buffer

        // Calculate the new downscaled dimensions
        val newWidth = image.width / scaleFactor
        val newHeight = image.height / scaleFactor

        // Allocate a much smaller byte array exactly the size of our target resolution
        val ySize = newWidth * newHeight
        val nv21 = ByteArray(ySize + (ySize / 2))

        val yRowStride = planes[0].rowStride
        val yPixelStride = planes[0].pixelStride

        var outIdx = 0

        // 1. Extract Y-Plane (Luminance/Grayscale) by skipping pixels
        for (row in 0 until newHeight) {
            val srcRow = row * scaleFactor
            for (col in 0 until newWidth) {
                val srcCol = col * scaleFactor
                val pos = (srcRow * yRowStride) + (srcCol * yPixelStride)
                nv21[outIdx++] = yBuffer.get(pos)
            }
        }

        val uRowStride = planes[1].rowStride
        val uPixelStride = planes[1].pixelStride
        val vRowStride = planes[2].rowStride
        val vPixelStride = planes[2].pixelStride

        // 2. Extract U and V planes (Color) - these are already half-resolution in YUV_420
        for (row in 0 until newHeight / 2) {
            val srcRow = row * scaleFactor
            for (col in 0 until newWidth / 2) {
                val srcCol = col * scaleFactor

                val uPos = (srcRow * uRowStride) + (srcCol * uPixelStride)
                val vPos = (srcRow * vRowStride) + (srcCol * vPixelStride)

                // NV21 format expects V then U interleaved
                nv21[outIdx++] = vBuffer.get(vPos)
                nv21[outIdx++] = uBuffer.get(uPos)
            }
        }

        // 3. Perform a SINGLE JPEG compression on the already-shrunken byte array
        val out = ByteArrayOutputStream()
        val yuvImage = YuvImage(nv21, ImageFormat.NV21, newWidth, newHeight, null)
        yuvImage.compressToJpeg(Rect(0, 0, newWidth, newHeight), 80, out)

        return out.toByteArray()
    }
}