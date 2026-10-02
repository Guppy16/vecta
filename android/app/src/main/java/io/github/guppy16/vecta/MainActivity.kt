package io.github.guppy16.vecta

import android.Manifest
import android.content.pm.PackageManager
import android.os.Bundle
import android.view.WindowManager
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
import io.github.guppy16.vecta.ui.VectaScreen

class MainActivity : ComponentActivity() {
    private var hasCamera by mutableStateOf(false)
    private val vm: SessionViewModel by viewModels()
    private val askPermissions =
        registerForActivityResult(ActivityResultContracts.RequestMultiplePermissions()) { granted ->
            hasCamera = granted[Manifest.permission.CAMERA] == true   // mic is optional: no audio track if denied
        }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)   // it's a live camera: never doze mid-task
        val wanted = arrayOf(Manifest.permission.CAMERA, Manifest.permission.RECORD_AUDIO)
        val missing = wanted.filter {
            ContextCompat.checkSelfPermission(this, it) != PackageManager.PERMISSION_GRANTED
        }
        hasCamera = Manifest.permission.CAMERA !in missing
        if (missing.isNotEmpty()) askPermissions.launch(missing.toTypedArray())   // ask for whatever's missing
        setContent {
            if (hasCamera) {
                VectaScreen(vm)   // the view model (and the camera) is created only once allowed
            } else {
                Box(Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
                    Text("Camera permission required.")
                }
            }
        }
    }

    // Backgrounded or screen off: stop streaming camera and mic, resume when visible again.
    override fun onStart() {
        super.onStart()
        if (hasCamera) vm.setForeground(true)
    }

    override fun onStop() {
        super.onStop()
        if (hasCamera) vm.setForeground(false)
    }
}
