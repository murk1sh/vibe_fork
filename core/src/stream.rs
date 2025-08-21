use crate::whisper::{self, WhisperContext, WhisperState};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{Device, SampleFormat, Stream, SupportedStreamConfig};
use crossbeam_channel::{unbounded, Receiver, Sender};
use eyre::{Context, Result};
use silero_vad::{Vad, VadIterator};
use std::sync::Arc;
use std::thread::{self, JoinHandle};

// --- Constants for Audio and VAD ---

// Whisper requires audio to be at a 16kHz sample rate.
const TARGET_SAMPLE_RATE: u32 = 16000;
// We'll use a VAD model optimized for this sample rate.
const VAD_MODEL_SAMPLE_RATE: i32 = 16000;
// The VAD processes audio in chunks. This size is recommended by the Silero VAD model.
const VAD_CHUNK_SIZE: usize = 512;
// VAD speech probability threshold.
const VAD_THRESHOLD: f32 = 0.5;
// How many milliseconds of silence to wait for before considering a phrase to be finished.
const SILENCE_THRESHOLD_MS: u32 = 500;

/// Manages the entire real-time transcription session.
pub struct LiveSession {
    // The cpal audio stream. Must be kept alive to continue capturing audio.
    _stream: Stream,
    // The handle to our dedicated Whisper processing thread.
    _whisper_thread: JoinHandle<()>,
}

impl LiveSession {
    /// Creates a new `LiveSession`, starting the audio capture and transcription process.
    ///
    /// # Arguments
    /// * `ctx` - A thread-safe Arc reference to the initialized WhisperContext.
    /// * `transcription_sender` - A channel sender to send final transcribed text back to the UI/main thread.
    ///
    /// # Returns
    /// A Result containing the new `LiveSession` or an error if setup fails.
    pub fn new(
        ctx: Arc<WhisperContext>,
        transcription_sender: Sender<String>,
    ) -> Result<Self> {
        // --- 1. Set up channels for communication between threads ---
        // This channel sends chunks of audio from the cpal audio thread to the Whisper thread.
        let (audio_sender, audio_receiver) = unbounded::<Vec<f32>>();

        // --- 2. Spawn a dedicated thread for Whisper processing ---
        // This is crucial to prevent the real-time audio callback from being blocked by
        // the computationally expensive transcription process.
        let whisper_thread = thread::spawn(move || {
            // Create a new state for this transcription session.
            let mut state = ctx
                .create_state()
                .expect("failed to create whisper state");

            // Loop forever, waiting for audio chunks from the audio thread.
            while let Ok(audio_chunk) = audio_receiver.recv() {
                println!(
                    "Whisper thread received audio chunk of size: {}",
                    audio_chunk.len()
                );

                // --- Run the transcription ---
                // We use the `full` method here on the single chunk of audio.
                // For more advanced use cases, you could manage a longer audio context.
                let params = whisper::setup_params_from_defaults(); // Use a default parameter setup
                state
                    .full(params, &audio_chunk)
                    .expect("failed to run model");

                // --- Extract and send the result ---
                let num_segments = state
                    .full_n_segments()
                    .expect("failed to get number of segments");

                if num_segments > 0 {
                    let mut full_text = String::new();
                    for i in 0..num_segments {
                        if let Ok(segment) = state.full_get_segment_text_lossy(i) {
                            full_text.push_str(&segment);
                        }
                    }
                    println!("Transcription: {}", full_text);
                    // Send the final text back to the main thread.
                    transcription_sender.send(full_text).unwrap();
                }
            }
        });

        // --- 3. Set up cpal for audio capture ---
        let host = cpal::default_host();
        let device = host
            .default_input_device()
            .context("No input device available")?;
        let config = find_supported_config(&device)?;

        // --- 4. Build and start the audio stream ---
        let stream = match config.sample_format() {
            SampleFormat::F32 => Self::build_stream::<f32>(device, config, audio_sender),
            SampleFormat::I16 => Self::build_stream::<i16>(device, config, audio_sender),
            SampleFormat::U16 => Self::build_stream::<u16>(device, config, audio_sender),
            _ => panic!("Unsupported sample format"),
        }?;

        stream.play().context("Failed to start audio stream")?;

        Ok(Self {
            _stream: stream,
            _whisper_thread: whisper_thread,
        })
    }

    /// Generic function to build a cpal stream for any supported sample type.
    fn build_stream<T: cpal::Sample + cpal::FromSample<f32>>(
        device: Device,
        config: SupportedStreamConfig,
        audio_sender: Sender<Vec<f32>>,
    ) -> Result<Stream> {
        // --- VAD and audio buffer setup ---
        let mut vad = Self::init_vad()?;
        let mut audio_buffer = Vec::new();
        let mut is_speaking = false;
        let silence_chunks_needed =
            (SILENCE_THRESHOLD_MS * (TARGET_SAMPLE_RATE / 1000) / VAD_CHUNK_SIZE as u32) as usize;
        let mut silence_counter = 0;

        let stream = device.build_input_stream(
            &config.into(),
            move |data: &[T], _: &cpal::InputCallbackInfo| {
                // Convert incoming audio data to the f32 format Whisper needs.
                let mut samples: Vec<f32> = data.iter().map(|s| s.to_sample()).collect();

                // --- Resample if necessary ---
                // This is a simplified resampling. For production, use a proper library like `rubato`.
                if config.sample_rate() != TARGET_SAMPLE_RATE {
                    // Simple resampling logic (can be improved)
                    let ratio = config.sample_rate() as f32 / TARGET_SAMPLE_RATE as f32;
                    let mut resampled = Vec::new();
                    let mut i = 0.0;
                    while i < samples.len() as f32 {
                        resampled.push(samples[i as usize]);
                        i += ratio;
                    }
                    samples = resampled;
                }

                // Process the audio in VAD-compatible chunks.
                for chunk in samples.chunks(VAD_CHUNK_SIZE) {
                    let probability = vad.predict(chunk.to_vec());
                    if probability > VAD_THRESHOLD {
                        // Speech detected
                        if !is_speaking {
                            println!("Speech started...");
                            is_speaking = true;
                            audio_buffer.clear(); // Start a new phrase
                        }
                        audio_buffer.extend_from_slice(chunk);
                        silence_counter = 0; // Reset silence counter
                    } else {
                        // Silence detected
                        if is_speaking {
                            silence_counter += 1;
                            if silence_counter > silence_chunks_needed {
                                println!("Speech ended.");
                                is_speaking = false;
                                // Send the collected audio buffer for transcription.
                                if !audio_buffer.is_empty() {
                                    audio_sender.send(audio_buffer.clone()).unwrap();
                                }
                            } else {
                                // Still within the silence threshold, so keep buffering.
                                audio_buffer.extend_from_slice(chunk);
                            }
                        }
                    }
                }
            },
            |err| eprintln!("An error occurred on the audio stream: {}", err),
            None,
        )?;
        Ok(stream)
    }

    /// Initializes the Silero VAD model.
    fn init_vad() -> Result<Vad> {
        let vad = Vad::builder()
            .sample_rate(VAD_MODEL_SAMPLE_RATE)
            .chunk_size(VAD_CHUNK_SIZE)
            .build()
            .context("Failed to build VAD")?;
        Ok(vad)
    }
}

/// Finds a supported input stream config, preferring our target sample rate.
fn find_supported_config(device: &Device) -> Result<SupportedStreamConfig> {
    let configs = device.supported_input_configs()?;
    configs
        .filter(|c| c.sample_format() == SampleFormat::F32 || c.sample_format() == SampleFormat::I16 || c.sample_format() == SampleFormat::U16)
        .find(|c| c.min_sample_rate().as_u32() <= TARGET_SAMPLE_RATE && TARGET_SAMPLE_RATE <= c.max_sample_rate().as_u32())
        .or_else(|| device.default_input_config().ok())
        .context("No supported input config found")
}
