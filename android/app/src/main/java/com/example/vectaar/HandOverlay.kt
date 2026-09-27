package com.example.vectaar

import androidx.compose.foundation.Canvas
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import org.json.JSONObject

/**
 * Hand overlay helpers for Vectaar.
 *
 * The server (vectaar_hand_server.py) sends per frame:
 *   {"state","raw_state","object_score","detected","predicted","landmarks":[[x,y]...]}
 * where landmarks are normalized [0,1] to the CAMERA IMAGE (not the screen).
 *
 * To draw them you must map IMAGE_NORMALIZED -> VIEW with the current ARCore
 * frame (same call you use for box placement) — see wiring notes in ARScreen.
 */

// MediaPipe 21-landmark hand topology.
val HAND_CONNECTIONS: List<Pair<Int, Int>> = listOf(
    0 to 1, 1 to 2, 2 to 3, 3 to 4,            // thumb
    0 to 5, 5 to 6, 6 to 7, 7 to 8,            // index
    5 to 9, 9 to 10, 10 to 11, 11 to 12,       // middle
    9 to 13, 13 to 14, 14 to 15, 15 to 16,     // ring
    13 to 17, 17 to 18, 18 to 19, 19 to 20,    // pinky
    0 to 17                                    // palm base
)

data class HandFrame(
    val state: String,
    val predicted: Boolean,
    val objectScore: Double?,        // grip non-skin value, or null
    val normLandmarks: FloatArray?   // [x0,y0,x1,y1,...] normalized to image, or null
)

/** Parse one server JSON message into a HandFrame. */
fun parseHandFrame(json: JSONObject): HandFrame {
    val lmArr = json.optJSONArray("landmarks")
    val norm = if (lmArr != null && lmArr.length() == 21) {
        FloatArray(42).also { a ->
            for (i in 0 until 21) {
                val p = lmArr.getJSONArray(i)
                a[i * 2] = p.getDouble(0).toFloat()
                a[i * 2 + 1] = p.getDouble(1).toFloat()
            }
        }
    } else null
    val obj = if (json.isNull("object_score")) null else json.optDouble("object_score")
    return HandFrame(
        state = json.optString("state", "no_hand"),
        predicted = json.optBoolean("predicted", false),
        objectScore = obj,
        normLandmarks = norm
    )
}

/** Class -> colour for the state label. */
fun stateColor(state: String): Color = when (state) {
    "grasp" -> Color(0xFF46C46A)   // green
    "point" -> Color(0xFFB574E0)   // purple
    "hand"  -> Color(0xFFE0A83C)   // amber
    else    -> Color(0xFF8A8F98)   // no_hand grey
}

/**
 * Draws the hand skeleton from VIEW-space pixel coords ([x0,y0,x1,y1,...], 42 floats).
 * Bones are green when the hand was detected this frame, orange when the Kalman
 * filter is predicting it through a detection gap.
 */
@Composable
fun HandSkeletonOverlay(
    viewPts: FloatArray?,
    detected: Boolean,
    modifier: Modifier = Modifier
) {
    if (viewPts == null || viewPts.size < 42) return
    val boneColor = if (detected) Color(0xFF00E676) else Color(0xFFFFA000)
    Canvas(modifier = modifier) {
        for ((a, b) in HAND_CONNECTIONS) {
            drawLine(
                color = boneColor,
                start = Offset(viewPts[a * 2], viewPts[a * 2 + 1]),
                end = Offset(viewPts[b * 2], viewPts[b * 2 + 1]),
                strokeWidth = 5f
            )
        }
        for (i in 0 until 21) {
            drawCircle(Color.Red, radius = 7f, center = Offset(viewPts[i * 2], viewPts[i * 2 + 1]))
        }
    }
}