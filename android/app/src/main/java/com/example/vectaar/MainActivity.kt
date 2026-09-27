package com.example.vectaar

import android.Manifest
import android.content.pm.PackageManager
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.activity.viewModels
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.material3.Text
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.core.content.ContextCompat
import com.example.vectaar.ui.CaptureScreen

class MainActivity : ComponentActivity() {
    private var hasCamera by mutableStateOf(false)
    private val askCamera = registerForActivityResult(ActivityResultContracts.RequestPermission()) {
        hasCamera = it
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        hasCamera = ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) ==
            PackageManager.PERMISSION_GRANTED
        if (!hasCamera) askCamera.launch(Manifest.permission.CAMERA)
        setContent {
            if (hasCamera) {
                val vm: SessionViewModel by viewModels()   // created only once we can open the camera
                CaptureScreen(vm)
            } else {
                Box(Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
                    Text("Camera permission required.")
                }
            }
        }
    }
}
