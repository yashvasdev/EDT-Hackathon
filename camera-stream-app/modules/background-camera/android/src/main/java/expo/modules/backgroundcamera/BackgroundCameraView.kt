package expo.modules.backgroundcamera

import android.content.Context
import android.graphics.Matrix
import android.graphics.SurfaceTexture
import android.view.Surface
import android.view.TextureView

import expo.modules.kotlin.AppContext
import expo.modules.kotlin.views.ExpoView

class BackgroundCameraView(context: Context, appContext: AppContext) : ExpoView(context, appContext) {

    private val textureView = TextureView(context)
    private var surface: Surface? = null

    private val module: BackgroundCameraModule?
        get() = appContext.registry.getModule()

    init {
        addView(textureView, LayoutParams(LayoutParams.MATCH_PARENT, LayoutParams.MATCH_PARENT))

        textureView.surfaceTextureListener = object : TextureView.SurfaceTextureListener {
            override fun onSurfaceTextureAvailable(st: SurfaceTexture, width: Int, height: Int) {
                st.setDefaultBufferSize(640, 480)
                applyRotationTransform(width, height)
                surface = Surface(st)
                module?.onPreviewSurfaceReady(surface!!)
            }

            override fun onSurfaceTextureSizeChanged(st: SurfaceTexture, width: Int, height: Int) {
                applyRotationTransform(width, height)
            }

            override fun onSurfaceTextureDestroyed(st: SurfaceTexture): Boolean {
                module?.onPreviewSurfaceDestroyed()
                surface?.release()
                surface = null
                return true
            }

            override fun onSurfaceTextureUpdated(st: SurfaceTexture) {}
        }
    }

    private fun applyRotationTransform(viewWidth: Int, viewHeight: Int) {
        val matrix = Matrix()
        val cx = viewWidth / 2f
        val cy = viewHeight / 2f
        // Rotate 90 degrees counter-clockwise to compensate for sensor orientation
        matrix.postRotate(-90f, cx, cy)
        // Scale to fill the view after rotation (swap aspect ratio)
        val scale = Math.max(
            viewWidth.toFloat() / viewHeight.toFloat(),
            viewHeight.toFloat() / viewWidth.toFloat()
        )
        matrix.postScale(scale, scale, cx, cy)
        textureView.setTransform(matrix)
    }
}
