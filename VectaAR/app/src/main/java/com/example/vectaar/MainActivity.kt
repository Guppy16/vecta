package com.example.vectaar

import android.Manifest
import android.content.pm.PackageManager
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.*
import androidx.compose.material3.Text
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.unit.dp
import androidx.core.content.ContextCompat

// --- NEW SCENEVIEW V4 & ARCORE IMPORTS ---
import com.google.ar.core.TrackingFailureReason
import com.google.ar.core.TrackingState
import io.github.sceneview.ar.ARSceneView

class MainActivity : ComponentActivity() {

    private val requestPermissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { isGranted: Boolean ->
        if (isGranted) {
            recreate()
        }
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

    Box(modifier = Modifier.fillMaxSize()) {
        // --- CHANGED TO ARSceneView FOR V4 ---
        ARSceneView(
            modifier = Modifier.fillMaxSize(),
            planeRenderer = true, // Shows dots on detected surfaces
            onSessionUpdated = { session, frame ->
                // Use native ARCore TrackingState
                val camera = frame.camera
                trackingStatus = when (camera.trackingState) {
                    TrackingState.TRACKING -> "Tracking: Good"
                    TrackingState.PAUSED -> {
                        val reason = camera.trackingFailureReason
                        "Tracking Paused: ${reason.name}"
                    }
                    TrackingState.STOPPED -> "Tracking Stopped"
                    else -> "Unknown"
                }
            }
        )

        // Simple debug UI
        Text(
            text = trackingStatus,
            color = Color.Green,
            modifier = Modifier
                .align(Alignment.BottomCenter)
                .padding(32.dp)
        )
    }
}
