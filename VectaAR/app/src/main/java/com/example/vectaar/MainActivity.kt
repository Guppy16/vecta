package com.example.vectaar
import com.example.vectaar.BuildConfig

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.ImageFormat
import android.graphics.Rect
import android.graphics.YuvImage
import android.media.Image
import android.os.Bundle
import android.util.Log
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.*
import androidx.compose.material3.Button
import androidx.compose.material3.Text
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.unit.dp
import androidx.core.content.ContextCompat
import com.google.ar.core.TrackingState
import io.github.sceneview.ar.ARSceneView
import okhttp3.*
import okio.ByteString.Companion.toByteString
import java.io.ByteArrayOutputStream
import java.io.IOException
import kotlin.time.Duration.Companion.seconds

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

    // Server configurations (Replace with your Python server IP)
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
                serverLogs = "Received: $text"
            }
            override fun onClosing(webSocket: WebSocket, code: Int, reason: String) {
                serverLogs = "Closing: $reason"
            }
            override fun onFailure(webSocket: WebSocket, t: Throwable, response: Response?) {
                serverLogs = "Error: ${t.localizedMessage}"
                isStreaming = false
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
                                val jpegBytes = ImageUtils.yuv420ToJpeg(image)
                                jpegBytes?.let {
                                    webSocket?.send(it.toByteString())
                                }
                            }
                        } catch (e: Exception) {
                            Log.e("ARStream", "Frame acquisition failed", e)
                        }
                    }
                }
            }
        )

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
    fun yuv420ToJpeg(image: Image): ByteArray? {
        if (image.format != ImageFormat.YUV_420_888) return null

        val planes = image.planes
        val yBuffer = planes[0].buffer
        val uBuffer = planes[1].buffer
        val vBuffer = planes[2].buffer

        val ySize = yBuffer.remaining()
        val uSize = uBuffer.remaining()
        val vSize = vBuffer.remaining()

        val nv21 = ByteArray(ySize + (image.width * image.height / 2))

        yBuffer.get(nv21, 0, ySize)

        val outOffset = ySize
        val uRowStride = planes[1].rowStride
        val vRowStride = planes[2].rowStride
        val uPixelStride = planes[1].pixelStride
        val vPixelStride = planes[2].pixelStride

        var outIdx = outOffset
        for (row in 0 until image.height / 2) {
            for (col in 0 until image.width / 2) {
                val uPos = row * uRowStride + col * uPixelStride
                val vPos = row * vRowStride + col * vPixelStride

                nv21[outIdx++] = vBuffer.get(vPos)
                nv21[outIdx++] = uBuffer.get(uPos)
            }
        }

        val out = ByteArrayOutputStream()
        val yuvImage = YuvImage(nv21, ImageFormat.NV21, image.width, image.height, null)

        yuvImage.compressToJpeg(Rect(0, 0, image.width, image.height), 75, out)
        return out.toByteArray()
    }
}