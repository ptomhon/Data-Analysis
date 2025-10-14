import numpy as np
import matplotlib.pyplot as plt

# --------------------------
# Load FID data from CSV
# --------------------------
def load_fid_data(file_path):
    """
    Load FID data from a CSV file where:
       - Column 1: Time in microseconds
       - Column 2: Real component
       - Column 3: Imaginary component
    """
    data = np.loadtxt(file_path, delimiter=',')
    time_arr = data[:, 0] / 1e6  # convert µs to seconds
    real_part = data[:, 1]
    imag_part = data[:, 2]
    fid = real_part + 1j * imag_part
    return time_arr, fid

# Load your FID data
time_arr, fid = load_fid_data(r"D:\WSU\Animal data may 2025 visit\copy of the 13C data\2025-05-07-animal injection coil #9\animal05\8\fid.csv")

# Parameters
target_length = 16384  # Target length for zero-filling
T2_apod = 5
dt = time_arr[1] - time_arr[0]

# Zero-filling and apodization
current_length = len(fid)
padded_signal = np.pad(fid, (0, target_length - current_length), mode='constant')
t = np.arange(target_length) * dt
apodization_filter = np.exp(-T2_apod * t)
processed_signal = padded_signal * apodization_filter

# FFT
fft_signal = np.fft.fft(processed_signal)
fft_shifted = np.fft.fftshift(fft_signal)
freqs = np.fft.fftfreq(target_length, d=dt)
freqs_shifted = np.fft.fftshift(freqs)

# Convert to ppm with 3.747 MHz reference frequency
ppm_axis = freqs_shifted / 3.828515
ppm_axis = ppm_axis[::-1]
spectrum_real = fft_shifted.real[::-1]

# Plot both time and frequency domain
plt.figure(figsize=(12, 6))

# Time-domain plot
plt.subplot(1, 2, 1)
plt.plot(t, processed_signal.real)
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.title("Time-domain FID")
plt.grid(True)

# Frequency-domain plot
plt.subplot(1, 2, 2)
plt.plot(ppm_axis, spectrum_real)
plt.xlabel("Chemical Shift (ppm)")
plt.ylabel("Amplitude")
plt.title("Frequency-domain Spectrum")
plt.gca().invert_xaxis()
plt.grid(True)

plt.tight_layout()
plt.show()
