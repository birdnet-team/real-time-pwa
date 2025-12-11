/**
 * BirdNET Live - Audio Processor Worklet
 * Replaces the deprecated ScriptProcessorNode.
 * Buffers audio frames and sends them to the main thread via MessagePort.
 */

class AudioProcessor extends AudioWorkletProcessor {
    constructor() {
        super();
        // Use a buffer size similar to what we had (2048) to avoid flooding the main thread
        this.bufferSize = 2048;
        this._buffer = new Float32Array(this.bufferSize);
        this._index = 0;
    }

    process(inputs, outputs, parameters) {
        // inputs[0] is the first input, which can have multiple channels
        const input = inputs[0];

        // We expect mono input usually, but we'll take channel 0
        if (input && input.length > 0) {
            const channelData = input[0];

            // Append to our internal buffer
            for (let i = 0; i < channelData.length; i++) {
                this._buffer[this._index++] = channelData[i];

                // When buffer is full, flush to main thread
                if (this._index >= this.bufferSize) {
                    this.port.postMessage(this._buffer);
                    this._index = 0;
                }
            }
        }

        return true; // Keep the processor alive
    }
}

registerProcessor('audio-processor', AudioProcessor);
