import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import hankel, svd, eig, lstsq

# %% 1. Signal Generation
# -------------------------
# Define parameters
f1 = 400
f2 = 350
T2 = 0.5
a1 = 10
a2 = 20
dt = 0.001

# Time vector: from 0 to 0.998 in steps of dt (999 points)
t = np.arange(0, 0.998 + dt, dt)

# Define the damped oscillatory functions for the real and imaginary parts.
def f_real(x):
    return (a1 * np.cos(2 * np.pi * f1 * x) + a2 * np.cos(2 * np.pi * f2 * x)) * np.exp(-x / T2)

def f_imag(x):
    return (a1 * np.sin(2 * np.pi * f1 * x) + a2 * np.sin(2 * np.pi * f2 * x)) * np.exp(-x / T2)

# Generate the clean complex signal (FID)
signal1 = f_real(t) + 1j * f_imag(t)

# Plot the clean complex FID: overlay real and imaginary parts
plt.figure(figsize=(8, 3))
plt.plot(t, signal1.real, label='Real Part')
plt.plot(t, signal1.imag, label='Imaginary Part')
plt.title("Clean Complex Signal (FID)")
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.legend()
plt.show()

# %% 2. Add Noise
# -------------------------
# Generate complex Gaussian noise: separate noise for real and imaginary parts.
noise_real = np.random.normal(0, 10, size=signal1.shape)
noise_imag = np.random.normal(0, 10, size=signal1.shape)
noise = noise_real + 1j * noise_imag

# Add noise to the clean signal
signal2 = signal1 + noise

# Plot the noisy complex FID: overlay real and imaginary parts
plt.figure(figsize=(8, 3))
plt.plot(t, signal2.real, label='Real Part')
plt.plot(t, signal2.imag, label='Imaginary Part')
plt.title("Noisy Complex Signal (FID)")
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.legend()
plt.show()

# %% 3. Fourier Spectrum
# -------------------------
# Compute Fourier transforms for the clean and noisy signals.
n = len(signal1)
freqs = np.fft.fftfreq(n, d=dt)

# Take only the first half (positive frequencies) and display only the real part of the FFT.
spect_clean = np.fft.fft(signal1)[:n//2].real
spect_noisy = np.fft.fft(signal2)[:n//2].real
freqs = freqs[:n//2]

plt.figure(figsize=(8, 3))
plt.plot(freqs, spect_clean)
plt.xlim(300, 450)
plt.ylim(-40, 110)
plt.title("Spectrum of Clean Signal (Real FFT Part)")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Amplitude")
plt.show()

plt.figure(figsize=(8, 3))
plt.plot(freqs, spect_noisy)
plt.xlim(300, 450)
plt.ylim(-40, 110)
plt.title("Spectrum of Noisy Signal (Real FFT Part)")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Amplitude")
plt.show()

# %% 4. Hankel Matrix Construction & SVD
# -------------------------
# Construct Hankel matrices using the clean and noisy complex FIDs.
v_clean = signal1  # length 999
v_noisy = signal2

# Build the Hankel matrix: for a square matrix of size 500x500 (500 + 500 - 1 = 999)
H_clean = hankel(v_clean[:500], v_clean[499:])
H_noisy = hankel(v_noisy[:500], v_noisy[499:])

# Compute the SVD for the clean and noisy Hankel matrices.
U_clean, s_clean, Vh_clean = svd(H_clean, full_matrices=False)
U_noisy, s_noisy, Vh_noisy = svd(H_noisy, full_matrices=False)

# %% 5. Iterative Hankel Projection via Anti‐diagonal Averaging
# -------------------------
# Retain k=4 singular values/vectors.
k = 4

# Helper function: average over anti-diagonals to reimpose Hankel structure.
def average_anti_diagonals(M):
    N, Mdim = M.shape
    M_new = np.copy(M)
    # For a square matrix, the index sum d = i+j runs from 0 to 2*N-2.
    for d in range(0, 2 * N - 1):
        # Find indices (i,j) such that i+j = d.
        indices = [(i, d - i) for i in range(max(0, d - (N - 1)), min(d + 1, N))]
        if indices:
            avg = np.mean([M[i, j] for (i, j) in indices])
            for (i, j) in indices:
                M_new[i, j] = avg
    return M_new

# Start with the noisy Hankel matrix as the initial guess.
newsig5 = np.copy(H_noisy)

# Iteratively perform SVD truncation (keeping k components) and reimpose the Hankel structure.
for l in range(6):
    U, s, Vh = svd(newsig5, full_matrices=False)
    U_trunc = U[:, :k]
    s_trunc = s[:k]
    Vh_trunc = Vh[:k, :]
    # Reconstruct the matrix from the truncated SVD.
    newsigint = U_trunc @ np.diag(s_trunc) @ Vh_trunc
    # Reimpose the Hankel structure by averaging along anti-diagonals.
    newsig5 = average_anti_diagonals(newsigint)

# %% 6. Reconstruct 1D Signal from the Hankel Matrix
# -------------------------
# Diagonal (Hankel) averaging to form the reconstructed signal.
first_column = newsig5[:, 0]
last_row = newsig5[1:, -1]  # exclude first element
newsignal1 = np.concatenate([first_column, last_row])

plt.figure(figsize=(8, 3))
plt.plot(np.arange(len(newsignal1)), newsignal1.real, label='Real Part')
plt.plot(np.arange(len(newsignal1)), newsignal1.imag, label='Imaginary Part')
plt.title("Reconstructed Signal (Diagonal Averaging)")
plt.xlabel("Sample")
plt.ylabel("Amplitude")
plt.legend()
plt.show()

# %% 7. Spectrum of the Reconstructed Signal
# -------------------------
n_new = len(newsignal1)
freqs_new = np.fft.fftfreq(n_new, d=dt)[:n_new//2]
spect_new = np.fft.fft(newsignal1)[:n_new//2].real  # display only the real part

plt.figure(figsize=(8, 3))
plt.plot(freqs_new, spect_new)
plt.xlim(300, 500)
plt.ylim(-30, 100)
plt.title("Spectrum of Reconstructed Signal (Real FFT Part)")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Amplitude")
plt.show()

# %% 8. Parameter Extraction via Eigenanalysis
# -------------------------
# Use the U matrix from the last SVD (after refinement) for eigenanalysis.
U5 = U_trunc  # shape: (500, k)

# Build two submatrices: one “shifted up” (Uf) and one “shifted down” (Ul).
Uf = U5[1:, :]
Ul = U5[:-1, :]

# Solve the least-squares problem: Ul * M ≈ Uf.
M_matrix = np.linalg.lstsq(Ul, Uf, rcond=None)[0]

# Compute the eigenvalues of M.
eigvals = np.linalg.eigvals(M_matrix)

# Estimate frequencies and decay times from the eigenvalues.
freqs_est = -1 / (2 * np.pi * dt) * np.angle(eigvals)
T2_est = -dt / np.log(np.abs(eigvals))

print("Estimated Frequencies (Hz):", freqs_est)
print("Estimated Decay Times:", T2_est)

# %% 9. Amplitude and Phase Estimation using a Vandermonde Matrix
# -------------------------
# Build a Vandermonde matrix A of size (999, k) from the eigenvalues.
N_signal = 999
A = np.vander(eigvals, N_signal, increasing=True).T  # Shape (999, k)

# Solve for the coefficients x in the least-squares sense.
x_coeff, residuals, rank, s_vals = np.linalg.lstsq(A, newsignal1, rcond=None)

# Extract the phase and amplitude for each mode.
phases = np.angle(x_coeff)
amps = np.abs(x_coeff)

print("Estimated Amplitudes:", amps)
print("Estimated Phases (radians):", phases)
