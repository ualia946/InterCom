import logging
import numpy as np
import minimal
import buffer
import sys
import threading
import time
from scipy.signal import correlate

minimal.parser.add_argument('--cancel_echo', action='store_true', help='Enable the echo cancellation feature.')

class EchoCancellation(buffer.Buffering):
    def __init__(self):
        super().__init__()
        
        self.PROBE_PULSE = self.generate_probe_pulse()
        self.PROBE_CAPTURE_DURATION_CHUNKS = 15
        self.probe_capture = None
        self.probing_counter = None
        
        self.playback_buffer = None
        self.delay = 0
        self.attenuation = 0.8
        self.is_calibrated = False
        self.calibration_thread = None
        self.lock = threading.Lock()
        
        logging.info("Echo Cancellation class initialized for Active Probing.")

    def generate_probe_pulse(self):
        chunk_size = minimal.args.frames_per_chunk
        num_channels = minimal.args.number_of_channels
        pulse = np.zeros((chunk_size, num_channels), dtype=np.int16)
        pulse[0, :] = 32767
        return pulse

    def _record_IO_and_play(self, ADC, DAC, frames, time, status):
        if not minimal.args.cancel_echo:
            super()._record_IO_and_play(ADC, DAC, frames, time, status)
            return

        if not self.is_calibrated and self.probing_counter is None:
            logging.info("Starting active calibration probe...")
            self.probing_counter = 0
            chunk_size = minimal.args.frames_per_chunk
            num_channels = minimal.args.number_of_channels
            self.probe_capture = np.zeros((chunk_size * self.PROBE_CAPTURE_DURATION_CHUNKS, num_channels), dtype=np.int16)

        if self.probing_counter is not None:
            chunk_size = minimal.args.frames_per_chunk
            
            if self.probing_counter == 0:
                DAC[:] = self.PROBE_PULSE
                logging.info("-> Probe pulse played locally.")
            else:
                DAC[:] = self.generate_zero_chunk()

            start = self.probing_counter * chunk_size
            end = start + chunk_size
            if end <= self.probe_capture.shape[0]:
                self.probe_capture[start:end, :] = ADC
            
            packed_chunk = self.pack(self.chunk_number, self.generate_zero_chunk())
            self.send(packed_chunk)
            self.chunk_number = (self.chunk_number + 1) % self.CHUNK_NUMBERS
            
            self.probing_counter += 1
            
            if self.probing_counter >= self.PROBE_CAPTURE_DURATION_CHUNKS:
                logging.info("-> Probe capture finished. Calculating parameters...")
                self.start_calibration(self.PROBE_PULSE, self.probe_capture)
                self.probing_counter = None
            return

        if self.is_calibrated:
            if self.playback_buffer is None:
                chunk_size = minimal.args.frames_per_chunk
                num_channels = minimal.args.number_of_channels
                self.playback_buffer = np.zeros((chunk_size * 20, num_channels), dtype=np.int16)

            chunk_size = minimal.args.frames_per_chunk
            num_channels = minimal.args.number_of_channels
            
            chunk_from_buffer = self.unbuffer_next_chunk().reshape(chunk_size, num_channels)
            self.playback_buffer = np.roll(self.playback_buffer, -chunk_size, axis=0)
            self.playback_buffer[-chunk_size:, :] = chunk_from_buffer

            eco_estimado = np.zeros_like(ADC, dtype=np.int16)
            with self.lock:
                if self.delay > 0:
                    eco_estimado = (np.roll(self.playback_buffer, self.delay, axis=0)[-chunk_size:, :] * self.attenuation).astype(np.int16)

            señal_limpia = np.clip(ADC.astype(np.int32) - eco_estimado.astype(np.int32), -32768, 32767).astype(np.int16)
            
            packed_chunk = self.pack(self.chunk_number, señal_limpia)
            self.send(packed_chunk)
            
            self.play_chunk(DAC, chunk_from_buffer)
            self.chunk_number = (self.chunk_number + 1) % self.CHUNK_NUMBERS

    def start_calibration(self, sent_pulse, captured_echo):
        if self.calibration_thread and self.calibration_thread.is_alive():
            return
        self.calibration_thread = threading.Thread(target=self.estimate_echo_parameters, args=(sent_pulse, captured_echo))
        self.calibration_thread.start()

    def estimate_echo_parameters(self, sent_pulse, captured_echo):
        try:
            pulse_mono = sent_pulse[:, 0]
            echo_mono = captured_echo[:, 0]
            
            correlation = correlate(echo_mono, pulse_mono, mode='full')
            
            # --- CORRECCIÓN CLAVE: IGNORAR EL CROSSTALK ---
            # Ignoramos los primeros 100 samples (~2ms) que es donde estaría el crosstalk.
            # Ponemos esa zona de la correlación a cero para que argmax no la elija.
            correlation_center = len(correlation) // 2
            ignore_samples = 100 
            correlation[correlation_center - ignore_samples : correlation_center + ignore_samples] = 0
            
            delay_index = np.argmax(np.abs(correlation)) - (len(pulse_mono) - 1)
            avg_delay = max(0, int(delay_index))

            peak_pulse = np.max(np.abs(pulse_mono))
            # Buscamos el pico del eco DESPUÉS del delay encontrado
            if avg_delay > 0:
                peak_echo = np.max(np.abs(echo_mono[avg_delay:]))
            else:
                peak_echo = 0

            attenuation = min(1.0, peak_echo / (peak_pulse + 1e-6)) if peak_pulse > 1e-6 else 1.0

            with self.lock:
                self.delay = avg_delay
                self.attenuation = attenuation
            
            logging.info(f"✅ Active calibration complete: Delay={self.delay} samples, Attenuation={self.attenuation:.2f}")

        except Exception as e:
            logging.error(f"Error during active calibration: {e}")
        finally:
            self.is_calibrated = True

# Clases Verbose... (sin cambios)
class EchoCancellationVerbose(EchoCancellation, buffer.Buffering__verbose):
    def __init__(self):
        super().__init__()

if __name__ == "__main__":
    minimal.parser.description = "Feedback Suppression Program"
    minimal.args = minimal.parser.parse_known_args()[0]

    if minimal.args.show_stats or minimal.args.show_samples or minimal.args.show_spectrum:
        intercom = EchoCancellationVerbose()
    else:
        intercom = EchoCancellation()

    try:
        intercom.run()
    except KeyboardInterrupt:
        minimal.parser.exit("\nSIGINT received")
    finally:
        intercom.print_final_averages()