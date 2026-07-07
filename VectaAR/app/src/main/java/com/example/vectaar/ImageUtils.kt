package com.example.vectaar

import android.graphics.ImageFormat
import android.graphics.Rect
import android.graphics.YuvImage
import android.media.Image
import androidx.camera.core.ImageProxy
import java.io.ByteArrayOutputStream

/**
 * Camera-frame helpers: pull YUV_420_888 planes off the ARCore image and encode
 * a downsampled JPEG. Stable — unchanged across the task-flow rework.
 */
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

    /** CameraX overload — same plane layout, different proxy type. */
    fun extractYuv(image: ImageProxy, timestamp: Long): Yuv420Frame {
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