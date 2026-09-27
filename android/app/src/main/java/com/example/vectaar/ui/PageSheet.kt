package com.example.vectaar.ui

import android.annotation.SuppressLint
import android.webkit.JavascriptInterface
import android.webkit.WebView
import android.webkit.WebViewClient
import androidx.compose.foundation.layout.fillMaxHeight
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.ModalBottomSheet
import androidx.compose.material3.rememberModalBottomSheetState
import androidx.compose.runtime.Composable
import androidx.compose.runtime.remember
import androidx.compose.ui.Modifier
import androidx.compose.ui.viewinterop.AndroidView
import com.example.vectaar.transport.PageRender
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonElement
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive

/**
 * The agent's page, slid up over the camera. Generated HTML is treated as
 * content: the app supplies base styling and the tap bridge, and JS the page
 * itself ships is limited to what the bridge exposes.
 */
@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun PageSheet(
    page: PageRender,
    baseUrl: String,
    onDismiss: () -> Unit,
    onEvent: (event: String, target: String?, data: JsonElement?) -> Unit,
) {
    val sheetState = rememberModalBottomSheetState(skipPartiallyExpanded = false)
    ModalBottomSheet(onDismissRequest = onDismiss, sheetState = sheetState) {
        AndroidView(
            modifier = Modifier.fillMaxWidth().fillMaxHeight(0.85f),
            factory = { ctx -> PageWebView(ctx, onEvent) },
            update = { web -> web.render(page, baseUrl) },
        )
    }
}

@SuppressLint("SetJavaScriptEnabled")
private class PageWebView(
    ctx: android.content.Context,
    private val onEvent: (String, String?, JsonElement?) -> Unit,
) : WebView(ctx) {
    private var shownVersion = -1

    init {
        settings.javaScriptEnabled = true
        settings.allowFileAccess = false
        settings.allowContentAccess = false
        webViewClient = object : WebViewClient() {
            override fun onPageFinished(view: WebView, url: String?) { evaluateJavascript(BRIDGE_JS, null) }
        }
        addJavascriptInterface(object {
            @JavascriptInterface
            fun send(json: String) {
                val obj = runCatching { Json.parseToJsonElement(json).jsonObject }.getOrNull() ?: return
                val event = obj["event"]?.jsonPrimitive?.content ?: return
                val target = (obj["target"] as? JsonPrimitive)?.takeIf { it.isString }?.content
                post { onEvent(event, target, obj["data"]) }
            }
        }, "vectaNative")
    }

    fun render(page: PageRender, baseUrl: String) {
        if (page.version == shownVersion) return
        shownVersion = page.version
        // baseUrl makes relative asset paths resolve against the server
        loadDataWithBaseURL(baseUrl, BASE_STYLE + page.html, "text/html", "utf-8", null)
    }
}

private const val BASE_STYLE = """<style>
:root{color-scheme:light dark}
body{margin:0;padding:16px;font:16px/1.4 system-ui,sans-serif;
  padding-bottom:calc(16px + env(safe-area-inset-bottom))}
img{max-width:100%;height:auto;border-radius:8px}
figure{margin:0 0 12px}figcaption{font-size:13px;opacity:.7}
h1{font-size:22px;margin:0 0 8px}code{font-size:13px}
[data-vecta-event]{cursor:pointer}
</style>"""

/** Taps on any element with data-vecta-event="…" are sent to the agent with its data-vecta-* attrs. */
private const val BRIDGE_JS = """
document.addEventListener('click', function (e) {
  var el = e.target.closest('[data-vecta-event]'); if (!el) return;
  var data = {};
  for (var a of el.attributes) if (a.name.startsWith('data-vecta-') && a.name !== 'data-vecta-event')
    data[a.name.slice(11)] = a.value;
  vectaNative.send(JSON.stringify({event: el.dataset.vectaEvent, target: el.dataset.vectaId || null, data: data}));
});
window.vecta = { send: function (event, data) {
  vectaNative.send(JSON.stringify({event: event, target: null, data: data || {}})); } };
"""
