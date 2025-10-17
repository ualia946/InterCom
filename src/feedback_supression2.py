import logging
import numpy as np
import minimal
import buffer
import sys
# CORRECCIÓN: Importamos el nombre de clase correcto
from speexdsp_ns import AcousticEchoCanceller

# 1. Añadimos el nuevo parámetro a la línea de comandos
minimal.parser.add_argument('--cancel_echo', action='store_true', help='Enable the echo cancellation feature.')

class EchoCancellation(buffer.Buffering):
    def __init__(self):
        super().__init__()
        
        # --- Inicialización del Cancelador de Eco Adaptativo ---
        self.aec = None
        if minimal.args.cancel_echo:
            rate = int(minimal.args.frames_per_second)
            chunk_size = minimal.args.frames_per_chunk
            
            # CORRECCIÓN: Usamos el nombre de clase correcto al crear el objeto
            self.aec = AcousticEchoCanceller(rate, chunk_size, filter_length_ms=150)
            logging.info("✅ Adaptive Echo Canceller (AEC) initialized.")

    def _record_IO_and_play(self, ADC, DAC, frames, time, status):
        if not minimal.args.cancel_echo or self.aec is None:
            super()._record_IO_and_play(ADC, DAC, frames, time, status)
            return

        # --- LÓGICA DE CANCELACIÓN DE ECO ADAPTATIVA ---
        
        chunk_size = minimal.args.frames_per_chunk
        num_channels = minimal.args.number_of_channels

        chunk_from_buffer = self.unbuffer_next_chunk().reshape(chunk_size, num_channels)
        
        mic_mono = ADC[:, 0].astype(np.int16)
        playback_mono = chunk_from_buffer[:, 0].astype(np.int16)
        
        # La magia ocurre aquí con el objeto aec
        mic_clean_mono = self.aec.cancel(mic_mono, playback_mono)
        
        señal_limpia = np.tile(mic_clean_mono.reshape(-1, 1), (1, num_channels))

        packed_chunk = self.pack(self.chunk_number, señal_limpia)
        self.send(packed_chunk)
        
        self.play_chunk(DAC, chunk_from_buffer)
        
        self.chunk_number = (self.chunk_number + 1) % self.CHUNK_NUMBERS

# Clases Verbose para depuración (no se modifican)
class EchoCancellationVerbose(EchoCancellation, buffer.Buffering__verbose):
    def __init__(self):
        super().__init__()

if __name__ == "__main__":
    minimal.parser.description = "Feedback Suppression Program with Adaptive Filter"
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