// AudioWorkletProcessor: runs on the real-time audio thread.
// Its only job is to forward raw 128-sample blocks to the main thread —
// no downsampling or math here, because blocking the audio thread causes
// glitches in *all* audio in the tab.
class MicCapture extends AudioWorkletProcessor {
  process(inputs) {
    const channel = inputs[0][0];
    if (channel) {
      // Copy: the engine reuses this buffer after we return.
      this.port.postMessage(new Float32Array(channel));
    }
    return true; // keep processor alive
  }
}
registerProcessor('mic-capture', MicCapture);
