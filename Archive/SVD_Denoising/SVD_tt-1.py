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

# Time vector: Note that in Mathematica, t goes from 0 to 0.998 in steps of dt.
t = np.arange(0, 0.998 + dt, dt)  # length 999

# Define the damped oscillatory function
def f(x):
    return (a1 * np.cos(2 * np.pi * f1 * x) + a2 * np.cos(2 * np.pi * f2 * x)) * np.exp(-x / T2)

# Generate the clean signal
signal1 = f(t)

# Plot the clean signal
plt.figure(figsize=(8, 3))
plt.plot(t, signal1)
plt.title("Clean Signal")
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.show()

# %% 2. Add Noise
# -------------------------
# Generate noise (Gaussian with mean 0 and sigma=10)
noise = np.random.normal(0, 10, size=signal1.shape)
signal2 = signal1 + noise

# Plot the noisy signal
plt.figure(figsize=(8, 3))
plt.plot(t, signal2)
plt.title("Noisy Signal")
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.show()

# %% 3. Fourier Spectrum
# -------------------------
# Compute Fourier transforms (using np.fft.fft) and corresponding frequencies.
n = len(signal1)
freqs = np.fft.fftfreq(n, d=dt)

# Take only the first half (positive frequencies)
spect_clean = np.fft.fft(signal1)[:n//2].real
# print(spect_clean[:10])
spect_noisy = np.fft.fft(signal2)[:n//2].real
# print(spect_noisy[:10])
freqs = freqs[:n//2]

plt.figure(figsize=(8, 3))
plt.plot(freqs, spect_clean)
plt.xlim(300, 450)
plt.ylim(-40, 110)
plt.title("Spectrum of Clean Signal")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Amplitude")
plt.show()

plt.figure(figsize=(8, 3))
plt.plot(freqs, spect_noisy)
plt.xlim(300, 450)
plt.ylim(-40, 110)
plt.title("Spectrum of Noisy Signal")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Amplitude")
plt.show()

# %% 4. Hankel Matrix Construction & SVD
# -------------------------
# In Mathematica, a Hankel matrix is built from the first 999 data points.
# For a square Hankel matrix of size 500×500, note that 500 + 500 – 1 = 999.
# Use the clean signal's amplitude vector.
print("Signal1 shape:", signal1.shape)  # (999,)
print(signal1[:10])
print("Signal2 shape:", signal2.shape)  # (999,)
print(signal2[:10])
v_clean = signal1  # length 999

# Create the Hankel matrix: first column from v_clean[0:500] and last row from v_clean[499:]
H_clean = hankel(v_clean[:500], v_clean[499:])

# Likewise for the noisy signal:
v_noisy = signal2
H_noisy = hankel(v_noisy[:500], v_noisy[499:])

# Compute the SVD for the clean and noisy Hankel matrices.
U_clean, s_clean, Vh_clean = svd(H_clean, full_matrices=False)
U_noisy, s_noisy, Vh_noisy = svd(H_noisy, full_matrices=False)

# %% 5. Iterative Hankel Projection via Anti‐diagonal Averaging
# -------------------------
# We will use k=4 (i.e. we retain 4 singular values/vectors)
k = 4

# Define a helper function to average over the anti-diagonals of a square matrix.
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

# We start with the noisy Hankel matrix (H_noisy) as our initial guess.
newsig5 = np.copy(H_noisy)

# Iteratively perform SVD truncation (keeping k components) and reimpose Hankel structure.
for l in range(5):
    # Compute the SVD
    U, s, Vh = svd(newsig5, full_matrices=False)
    # Truncate to the first k singular values/vectors.
    U_trunc = U[:, :k]
    s_trunc = s[:k]
    Vh_trunc = Vh[:k, :]
    # Reconstruct the matrix from the truncated SVD.
    newsigint = U_trunc @ np.diag(s_trunc) @ Vh_trunc
    # Reimpose the Hankel structure by averaging along anti-diagonals.
    newsig5 = average_anti_diagonals(newsigint)

# %% 6. Reconstruct 1D Signal from the Hankel Matrix
# -------------------------
# Diagonal (or “Hankel”) averaging: In Mathematica the new signal is formed as:
# Flatten[Append[newsig5[[All,1]], newsig5[[2;;500,500]]]];
# That is, concatenate the first column of newsig5 with the last row (excluding its first element).
first_column = newsig5[:, 0]
last_row = newsig5[1:, -1]  # rows 1 to end of the last column
newsignal1 = np.concatenate([first_column, last_row])

plt.figure(figsize=(8, 3))
plt.plot(np.arange(len(newsignal1)), newsignal1)
plt.title("Reconstructed Signal (Diagonal Averaging)")
plt.xlabel("Sample")
plt.ylabel("Amplitude")
plt.show()

# %% 7. Spectrum of the Reconstructed Signal
# -------------------------
n_new = len(newsignal1)
freqs_new = np.fft.fftfreq(n_new, d=dt)[:n_new//2]
spect_new = np.fft.fft(newsignal1)[:n_new//2].real

plt.figure(figsize=(8, 3))
plt.plot(freqs_new, spect_new)
plt.xlim(300, 500)
plt.ylim(-30, 100)
plt.title("Spectrum of Reconstructed Signal")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Amplitude")
plt.show()

# %% 8. Parameter Extraction via Eigenanalysis
# -------------------------
# In the Mathematica code, a matrix M is estimated from a shifted version of the SVD “left singular vectors.”
# Here we use the U matrix from the last SVD (after refinement). Assume U5 = U_trunc (from the last iteration)
U5 = U_trunc  # shape: (500, k)

# Build two submatrices: one “shifted up” and one “shifted down.”
Uf = U5[1:, :]  # rows 1 to end (Mathematica: from 2nd row onward)
Ul = U5[:-1, :] # rows 0 to end-1

# Solve the least-squares problem for M: Ul * M ≈ Uf.
M_matrix = np.linalg.lstsq(Ul, Uf, rcond=None)[0]

# Compute the eigenvalues of M.
eigvals = np.linalg.eigvals(M_matrix)

# Estimate the frequencies and decay times from these eigenvalues.
# Frequencies: f = -1/(2*pi*dt) * Arg(eigenvalue)
freqs_est = -1 / (2 * np.pi * dt) * np.angle(eigvals)
# Decay times: T2 = -dt / log(|eigenvalue|)
T2_est = -dt / np.log(np.abs(eigvals))

print("Estimated Frequencies (Hz):", freqs_est)
print("Estimated Decay Times:", T2_est)

# %% 9. Amplitude and Phase Estimation using a Vandermonde Matrix
# -------------------------
# Build a Vandermonde matrix A of size (999, k) from the eigenvalues.
N_signal = 999
# Each column j is: [1, eigvals[j], eigvals[j]**2, ..., eigvals[j]**(N_signal-1)]
# np.vander with increasing=True gives exactly that.
A = np.vander(eigvals, N_signal, increasing=True).T  # Shape (999, k)

# Solve for the coefficients x in the least-squares sense: A * x ≈ newsignal1.
x_coeff, residuals, rank, s_vals = np.linalg.lstsq(A, newsignal1, rcond=None)

# Extract the phase and amplitude for each mode.
phases = np.angle(x_coeff)
amps = np.abs(x_coeff)

print("Estimated Amplitudes:", amps)
print("Estimated Phases (radians):", phases)
