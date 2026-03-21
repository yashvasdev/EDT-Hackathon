package expo.modules.backgroundcamera

import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import android.graphics.ImageFormat
import android.graphics.Rect
import android.graphics.YuvImage
import android.hardware.camera2.CameraCaptureSession
import android.hardware.camera2.CameraCharacteristics
import android.hardware.camera2.CameraDevice
import android.hardware.camera2.CameraManager
import android.hardware.camera2.CaptureRequest
import android.media.ImageReader
import android.os.Build
import android.os.Handler
import android.os.HandlerThread
import android.util.Base64
import androidx.core.content.ContextCompat

import expo.modules.kotlin.modules.Module
import expo.modules.kotlin.modules.ModuleDefinition

import java.io.ByteArrayOutputStream

class BackgroundCameraModule : Module() {

    private var cameraDevice: CameraDevice? = null
    private var captureSession: CameraCaptureSession? = null
    private var imageReader: ImageReader? = null
    private var backgroundThread: HandlerThread? = null
    private var backgroundHandler: Handler? = null
    private var isCapturing = false
    private var lastFrameTime = 0L
    private var frameIntervalMs = 500L

    private val context: Context
        get() = requireNotNull(appContext.reactContext)

    override fun definition() = ModuleDefinition {

        Name("BackgroundCamera")

        Events("onFrame", "onError")

        AsyncFunction("checkConcurrentSupport") {
            val cameraManager = context.getSystemService(Context.CAMERA_SERVICE) as CameraManager

            if (Build.VERSION.SDK_INT < 30) {
                return@AsyncFunction mapOf(
                    "supported" to false,
                    "reason" to "Requires Android 11+ (API 30). Device is API ${Build.VERSION.SDK_INT}."
                )
            }

            val concurrentIds = cameraManager.concurrentCameraIds
            if (concurrentIds.isEmpty()) {
                return@AsyncFunction mapOf(
                    "supported" to false,
                    "reason" to "Device camera HAL does not report any concurrent camera pairs."
                )
            }

            // Find a pair that contains both a front and back camera
            var frontId: String? = null
            var backId: String? = null

            for (idSet in concurrentIds) {
                var setFront: String? = null
                var setBack: String? = null
                for (id in idSet) {
                    val chars = cameraManager.getCameraCharacteristics(id)
                    val facing = chars.get(CameraCharacteristics.LENS_FACING)
                    if (facing == CameraCharacteristics.LENS_FACING_FRONT) setFront = id
                    if (facing == CameraCharacteristics.LENS_FACING_BACK) setBack = id
                }
                if (setFront != null && setBack != null) {
                    frontId = setFront
                    backId = setBack
                    break
                }
            }

            if (frontId != null && backId != null) {
                return@AsyncFunction mapOf(
                    "supported" to true,
                    "reason" to "Concurrent front ($frontId) + back ($backId) cameras supported.",
                    "frontCameraId" to frontId,
                    "backCameraId" to backId
                )
            }

            return@AsyncFunction mapOf(
                "supported" to false,
                "reason" to "Concurrent camera pairs exist, but no front+back pair found."
            )
        }

        AsyncFunction("startCapture") { intervalMs: Int ->
            if (isCapturing) {
                return@AsyncFunction mapOf("started" to false, "reason" to "Already capturing.")
            }

            if (ContextCompat.checkSelfPermission(context, Manifest.permission.CAMERA)
                != PackageManager.PERMISSION_GRANTED
            ) {
                return@AsyncFunction mapOf("started" to false, "reason" to "Camera permission not granted.")
            }

            frameIntervalMs = intervalMs.toLong()
            startBackgroundThread()

            val cameraManager = context.getSystemService(Context.CAMERA_SERVICE) as CameraManager

            // Find front camera
            var frontCameraId: String? = null
            for (id in cameraManager.cameraIdList) {
                val chars = cameraManager.getCameraCharacteristics(id)
                if (chars.get(CameraCharacteristics.LENS_FACING) == CameraCharacteristics.LENS_FACING_FRONT) {
                    frontCameraId = id
                    break
                }
            }

            if (frontCameraId == null) {
                return@AsyncFunction mapOf("started" to false, "reason" to "No front camera found.")
            }

            try {
                openCamera(cameraManager, frontCameraId)
                return@AsyncFunction mapOf("started" to true, "reason" to "Front camera capture started.")
            } catch (e: Exception) {
                sendEvent("onError", mapOf("message" to "Failed to open camera: ${e.message}"))
                return@AsyncFunction mapOf("started" to false, "reason" to "Failed to open camera: ${e.message}")
            }
        }

        AsyncFunction("stopCapture") {
            cleanup()
            return@AsyncFunction true
        }

        OnDestroy {
            cleanup()
        }
    }

    @Suppress("MissingPermission")
    private fun openCamera(cameraManager: CameraManager, cameraId: String) {
        imageReader = ImageReader.newInstance(640, 480, ImageFormat.YUV_420_888, 2).apply {
            setOnImageAvailableListener({ reader ->
                val now = System.currentTimeMillis()
                if (now - lastFrameTime < frameIntervalMs) {
                    reader.acquireLatestImage()?.close()
                    return@setOnImageAvailableListener
                }
                lastFrameTime = now

                val image = reader.acquireLatestImage() ?: return@setOnImageAvailableListener
                try {
                    val base64 = yuvToBase64Jpeg(image, 5)
                    sendEvent("onFrame", mapOf(
                        "base64" to base64,
                        "timestamp" to now
                    ))
                } catch (e: Exception) {
                    sendEvent("onError", mapOf("message" to "Frame encode error: ${e.message}"))
                } finally {
                    image.close()
                }
            }, backgroundHandler)
        }

        cameraManager.openCamera(cameraId, object : CameraDevice.StateCallback() {
            override fun onOpened(camera: CameraDevice) {
                cameraDevice = camera
                createCaptureSession()
            }

            override fun onDisconnected(camera: CameraDevice) {
                camera.close()
                cameraDevice = null
                isCapturing = false
            }

            override fun onError(camera: CameraDevice, error: Int) {
                camera.close()
                cameraDevice = null
                isCapturing = false
                sendEvent("onError", mapOf("message" to "Camera device error: $error"))
            }
        }, backgroundHandler)
    }

    private fun createCaptureSession() {
        val device = cameraDevice ?: return
        val reader = imageReader ?: return

        device.createCaptureSession(
            listOf(reader.surface),
            object : CameraCaptureSession.StateCallback() {
                override fun onConfigured(session: CameraCaptureSession) {
                    captureSession = session
                    isCapturing = true

                    val request = device.createCaptureRequest(CameraDevice.TEMPLATE_PREVIEW).apply {
                        addTarget(reader.surface)
                    }.build()

                    session.setRepeatingRequest(request, null, backgroundHandler)
                }

                override fun onConfigureFailed(session: CameraCaptureSession) {
                    isCapturing = false
                    sendEvent("onError", mapOf("message" to "Capture session configuration failed."))
                }
            },
            backgroundHandler
        )
    }

    private fun yuvToBase64Jpeg(image: android.media.Image, quality: Int): String {
        val yBuffer = image.planes[0].buffer
        val uBuffer = image.planes[1].buffer
        val vBuffer = image.planes[2].buffer

        val ySize = yBuffer.remaining()
        val uSize = uBuffer.remaining()
        val vSize = vBuffer.remaining()

        val nv21 = ByteArray(ySize + uSize + vSize)
        yBuffer.get(nv21, 0, ySize)
        vBuffer.get(nv21, ySize, vSize)
        uBuffer.get(nv21, ySize + vSize, uSize)

        val yuvImage = YuvImage(nv21, ImageFormat.NV21, image.width, image.height, null)
        val out = ByteArrayOutputStream()
        yuvImage.compressToJpeg(Rect(0, 0, image.width, image.height), quality, out)
        return Base64.encodeToString(out.toByteArray(), Base64.NO_WRAP)
    }

    private fun startBackgroundThread() {
        backgroundThread = HandlerThread("BackgroundCameraThread").also { it.start() }
        backgroundHandler = Handler(backgroundThread!!.looper)
    }

    private fun cleanup() {
        isCapturing = false
        captureSession?.close()
        captureSession = null
        cameraDevice?.close()
        cameraDevice = null
        imageReader?.close()
        imageReader = null
        backgroundThread?.quitSafely()
        backgroundThread = null
        backgroundHandler = null
    }
}
