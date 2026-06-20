package com.example.vecta

import android.Manifest
import android.content.pm.PackageManager
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.core.CameraSelector
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.compose.foundation.layout.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.lifecycle.compose.LocalLifecycleOwner
import androidx.compose.ui.unit.dp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import com.example.vecta.ui.theme.VectaTheme
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import android.graphics.Bitmap
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import okhttp3.WebSocket
import okio.ByteString.Companion.toByteString
import java.io.ByteArrayOutputStream
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.WebSocketListener
import androidx.core.graphics.createBitmap
import androidx.core.graphics.scale
import android.widget.Toast
import androidx.compose.runtime.LaunchedEffect
import okhttp3.Response
import androidx.compose.foundation.background
import androidx.compose.ui.graphics.Color
import okhttp3.Call
import okhttp3.Callback
import java.io.IOException
import android.util.Size
import androidx.camera.core.resolutionselector.ResolutionSelector
import androidx.camera.core.resolutionselector.ResolutionStrategy
import androidx.activity.compose.setContent
import androidx.compose.foundation.background
//import androidx.compose.layout.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import androidx.core.graphics.createBitmap
import androidx.lifecycle.compose.LocalLifecycleOwner
import okhttp3.*
import okio.ByteString.Companion.toByteString
import java.nio.ByteBuffer

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent {
            MaterialTheme {
                Surface(modifier = Modifier.fillMaxSize(), color = MaterialTheme.colorScheme.background) {
                    CameraPreviewScreen()
                }
            }
        }
    }
}

class FrameAnalyzer : ImageAnalysis.Analyzer {
    var webSocket: WebSocket? = null
    var currentEchoTimestamp: Long = 0L // Hold the laptop's clock token
    private var lastAnalyzedTimestamp = 0L

    override fun analyze(image: ImageProxy) {
        val currentTimestamp = System.currentTimeMillis()

        // Sample frame every 100ms if connected
        if (currentTimestamp - lastAnalyzedTimestamp >= 100 && webSocket != null) {
            val bitmap = image.toBitmap()
            val outputStream = ByteArrayOutputStream()
            bitmap.compress(Bitmap.CompressFormat.JPEG, 60, outputStream)
            val jpegBytes = outputStream.toByteArray()

            val buffer = ByteBuffer.allocate(8 + jpegBytes.size)
            // Prepend the ECHOD LAPTOP TIMESTAMP instead of the phone's clock
            buffer.putLong(currentEchoTimestamp)
            buffer.put(jpegBytes)

            webSocket?.send(buffer.array().toByteString())
            lastAnalyzedTimestamp = currentTimestamp
        }
        image.close()
    }
}

@Composable
fun CameraPreviewScreen() {
    val context = LocalContext.current
    val lifecycleOwner = LocalLifecycleOwner.current
    val cameraProviderFuture = remember { ProcessCameraProvider.getInstance(context) }

    var statusText by remember { mutableStateOf("Not Connected.") }
    val frameAnalyzer = remember { FrameAnalyzer() }

    // UI control states
    var isCameraActive by remember { mutableStateOf(true) }
    var activeWebSocket by remember { mutableStateOf<WebSocket?>(null) }

    // CRITICAL: Ensure this matches your exact laptop local IP
    val laptopIp = "192.168.0.81"

    Box(modifier = Modifier.fillMaxSize()) {
        if (isCameraActive) {
            AndroidView(
                factory = { ctx -> PreviewView(ctx).apply { scaleType = PreviewView.ScaleType.FILL_CENTER } },
                modifier = Modifier.fillMaxSize(),
                update = { previewView ->
                    cameraProviderFuture.addListener({
                        val cameraProvider = cameraProviderFuture.get()
                        val preview = Preview.Builder().build().also { it.surfaceProvider = previewView.surfaceProvider }
                        val resolutionSelector = ResolutionSelector.Builder()
                            .setResolutionStrategy(
                                ResolutionStrategy(
//                                    Size(1920, 1080), // Perfect balance of speed and AI visibility
                                    Size(640, 480), // Perfect balance of speed and AI visibility
                                    ResolutionStrategy.FALLBACK_RULE_CLOSEST_HIGHER_THEN_LOWER
                                )
                            ).build()
                        val imageAnalysis = ImageAnalysis.Builder()
                            .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                            .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                            .setResolutionSelector(resolutionSelector) // <-- Replaces setTargetResolution
                            .build()
                            .also { it.setAnalyzer(ContextCompat.getMainExecutor(context), frameAnalyzer) }
//                        val imageAnalysis = ImageAnalysis.Builder()
//                            .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
//                            .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
//                            // PUSH HARDWARE TO FULL 1080p
//                            .setTargetResolution(android.util.Size(1920, 1080))
//                            .build()
//                            .also { it.setAnalyzer(ContextCompat.getMainExecutor(context), frameAnalyzer) }
//                        val imageAnalysis = ImageAnalysis.Builder()
//                            .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
//                            .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
//                            .build()
//                            .also { it.setAnalyzer(ContextCompat.getMainExecutor(context), frameAnalyzer) }

                        try {
                            cameraProvider.unbindAll()
                            cameraProvider.bindToLifecycle(lifecycleOwner, CameraSelector.DEFAULT_BACK_CAMERA, preview, imageAnalysis)
                        } catch (e: Exception) {
                            statusText = "Camera Error: ${e.localizedMessage}"
                        }
                    }, ContextCompat.getMainExecutor(context))
                }
            )
        } else {
            // Placeholder when camera is completely shut down
            Box(modifier = Modifier.fillMaxSize().background(Color.DarkGray), contentAlignment = Alignment.Center) {
                Text("Camera Lens Closed", color = Color.White, style = MaterialTheme.typography.headlineSmall)
            }
        }

        // Control HUD Dashboard
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .background(Color.Black.copy(alpha = 0.75f))
                .padding(16.dp)
                .align(Alignment.TopCenter),
            horizontalAlignment = Alignment.CenterHorizontally
        ) {
            Text("Target Address: $laptopIp:8000", color = Color.LightGray, style = MaterialTheme.typography.bodySmall)
            Text(
                text = "Status: $statusText",
                color = if (statusText.contains("Connected") || statusText.contains("success")) Color.Green else Color.Yellow,
                style = MaterialTheme.typography.bodyLarge,
                modifier = Modifier.padding(vertical = 4.dp)
            )

            // Row 1: Connection Utilities
            Row(horizontalArrangement = Arrangement.spacedBy(8.dp), modifier = Modifier.padding(bottom = 4.dp)) {
                Button(onClick = {
                    statusText = "Pinging..."
                    val client = OkHttpClient()
                    val request = Request.Builder().url("http://$laptopIp:8000/ping").build()
                    client.newCall(request).enqueue(object : Callback {
                        override fun onResponse(call: Call, response: Response) {
                            val body = response.body?.string() ?: ""
                            ContextCompat.getMainExecutor(context).execute { statusText = "Ping success! Server: $body" }
                        }
                        override fun onFailure(call: Call, e: IOException) {
                            ContextCompat.getMainExecutor(context).execute { statusText = "Ping failed: ${e.localizedMessage}" }
                        }
                    })
                }) {
                    Text("Ping")
                }

                Button(
                    onClick = {
                        statusText = "Connecting Stream..."
                        val client = OkHttpClient()
                        val request = Request.Builder().url("ws://$laptopIp:8000/ws").build()
                        client.newWebSocket(request, object : WebSocketListener() {
                            override fun onOpen(webSocket: WebSocket, response: Response) {
                                activeWebSocket = webSocket
                                frameAnalyzer.webSocket = webSocket
                                ContextCompat.getMainExecutor(context).execute { statusText = "Stream Connected!" }
                            }

                            override fun onMessage(webSocket: WebSocket, text: String) {
                                try {
                                    val laptopTime = text.toLong()
                                    frameAnalyzer.currentEchoTimestamp = laptopTime
                                } catch (e: Exception) {
                                    e.printStackTrace()
                                }
                            }

                            override fun onFailure(webSocket: WebSocket, t: Throwable, response: Response?) {
                                ContextCompat.getMainExecutor(context).execute { statusText = "WS Fail: ${t.localizedMessage}" }
                            }
                        })
                    },
                    enabled = activeWebSocket == null
                ) {
                    Text("Connect")
                }

                Button(
                    onClick = {
                        activeWebSocket?.close(1000, "User disconnected manual hook")
                        activeWebSocket = null
                        frameAnalyzer.webSocket = null
                        statusText = "Stream Disconnected."
                    },
                    enabled = activeWebSocket != null
                ) {
                    Text("Disconnect")
                }
            }

            // Row 2: Camera Lifecycle Hardware Control
            Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                Button(
                    colors = ButtonDefaults.buttonColors(containerColor = if (isCameraActive) MaterialTheme.colorScheme.error else MaterialTheme.colorScheme.primary),
                    onClick = {
                        if (isCameraActive) {
                            // Turn Off Camera Hardware completely
                            val cameraProvider = cameraProviderFuture.get()
                            cameraProvider.unbindAll()
                            isCameraActive = false
                            statusText = "Camera Closed."
                        } else {
                            // Reboot Camera Hardware
                            isCameraActive = true
                            statusText = "Camera Opened."
                        }
                    }
                ) {
                    Text(if (isCameraActive) "Close Camera" else "Open Camera")
                }
            }
        }
    }
}