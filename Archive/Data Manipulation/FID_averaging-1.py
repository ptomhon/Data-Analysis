import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

def sum_average_fids(basepath, n_folders=200):
    fids = []
    time = None
    
    for i in range(1, n_folders+1):
        fid_path = os.path.join(basepath, str(i), "fid.csv")
        if os.path.exists(fid_path):
            data = np.loadtxt(fid_path, delimiter=",", skiprows=1)
            if time is None:
                time = data[:,0]  # ms
            real = data[:,1]
            imag = data[:,2]
            complex_fid = real + 1j*imag
            fids.append(complex_fid)
        else:
            print(f"Warning: {fid_path} not found, skipping.")
    
    if not fids:
        raise ValueError("No fid.csv files found in the given path.")
    
    fid_avg = np.mean(fids, axis=0)
    return time, fid_avg

def apply_apodization(fid, time, lb_hz):
    """Apply exponential apodization (line broadening)."""
    t_sec = time * 1e-3  # convert ms to s
    return fid * np.exp(-lb_hz * t_sec * np.pi * 2 / (2*np.pi))  # simplifies to exp(-lb * t)

def phase_correct_spectrum(fid, time, phi0, phi1, lb_hz):
    """Apply apodization + FFT + phase correction."""
    # Apodization
    fid_proc = apply_apodization(fid, time, lb_hz)

    # FFT
    n = len(fid_proc)
    spectrum = np.fft.fftshift(np.fft.fft(fid_proc))
    freq = np.fft.fftshift(np.fft.fftfreq(n, d=(time[1]-time[0])*1e-3))  # Hz
    
    # Phase correction
    pivot = np.mean(freq)
    phase_rad = np.deg2rad(phi0 + phi1*(freq - pivot)/(freq[-1]-freq[0]))
    spectrum_phased = spectrum * np.exp(1j*phase_rad)

    # Convert to ppm axis
    ppm = -((freq - (2500 - 639.08016399999997)) / 15.507665)

    return ppm, spectrum, spectrum_phased

def interactive_plot(time, fid_avg):
    # Initial slider values
    phi0_init, phi1_init, lb_init = 0, 0, 0.0

    # Compute FFT
    ppm, spectrum, spectrum_phased = phase_correct_spectrum(fid_avg, time, phi0_init, phi1_init, lb_init)
    
    fig, (ax_fid, ax_fft) = plt.subplots(1,2, figsize=(12,5))
    plt.subplots_adjust(bottom=0.3)  # leave space for sliders

    # Time-domain FID
    (line_fid_real,) = ax_fid.plot(time, fid_avg.real, label="Real")
    (line_fid_imag,) = ax_fid.plot(time, fid_avg.imag, label="Imag", alpha=0.7)
    ax_fid.set_xlabel("Time (ms)")
    ax_fid.set_ylabel("Signal")
    ax_fid.set_title("Averaged FID")
    ax_fid.legend()

    # Frequency-domain spectrum (real part, ppm axis)
    (line_fft,) = ax_fft.plot(ppm, spectrum_phased.real, label="Real (phased)")
    ax_fft.set_xlabel("Chemical Shift (ppm)")
    ax_fft.set_ylabel("Real Amplitude")
    ax_fft.set_title("FFT with Phase + Apodization")
    ax_fft.invert_xaxis()  # conventional ppm axis
    ax_fft.legend()

    # Slider axes
    ax_phi0 = plt.axes([0.25, 0.2, 0.65, 0.03])
    ax_phi1 = plt.axes([0.25, 0.15, 0.65, 0.03])
    ax_lb   = plt.axes([0.25, 0.1, 0.65, 0.03])

    # Sliders
    slider_phi0 = Slider(ax_phi0, 'Phase 0°', -180, 180, valinit=phi0_init)
    slider_phi1 = Slider(ax_phi1, 'Phase 1°', -180, 180, valinit=phi1_init)
    slider_lb   = Slider(ax_lb, 'LB (Hz)', 0, 5, valinit=lb_init)

    def update(val):
        phi0 = slider_phi0.val
        phi1 = slider_phi1.val
        lb   = slider_lb.val
        ppm, spectrum, spectrum_phased = phase_correct_spectrum(fid_avg, time, phi0, phi1, lb)
        
        # Update spectrum
        line_fft.set_xdata(ppm)
        line_fft.set_ydata(spectrum_phased.real)
        ax_fft.relim()
        ax_fft.autoscale_view()

        fig.canvas.draw_idle()

    slider_phi0.on_changed(update)
    slider_phi1.on_changed(update)
    slider_lb.on_changed(update)

    plt.show()

if __name__ == "__main__":
    basepath = r"D:\WSU\Raw Data\Spinsolve-1.4T_13C\2025-10-24\251024-112402 7degCarbon-Cells (PYR8.75_1)"
    time, fid_avg = sum_average_fids(basepath, n_folders=100)
    interactive_plot(time, fid_avg)
