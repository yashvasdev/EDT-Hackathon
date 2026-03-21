package expo.modules.backgroundcamera

import android.content.Context
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
                surface = Surface(st)
                module?.onPreviewSurfaceReady(surface!!)
            }

            override fun onSurfaceTextureSizeChanged(st: SurfaceTexture, width: Int, height: Int) {}

            override fun onSurfaceTextureDestroyed(st: SurfaceTexture): Boolean {
                module?.onPreviewSurfaceDestroyed()
                surface?.release()
                surface = null
                return true
            }

            override fun onSurfaceTextureUpdated(st: SurfaceTexture) {}
        }
    }
}
