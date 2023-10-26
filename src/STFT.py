import torchaudio
import torch
import math
torch.pi = torch.acos(torch.zeros(1)).item() * 2

class ipcSTFT(torch.nn.Module):
    def __init__(
        self,
        n_fft,
        hop_length=None,
        win_length=None,
        window=None,
        center=True,
        pad_mode="reflect",
        normalized=False,
        onesided=None  
    ):
        """ Similar arguments to torch.stft(). Note: return_complex is always True. """
        self.n_fft = n_fft
        self.hop_length = hop_length if hop_length is not None \
            else math.floor(self.n_fft / 4)
        self.win_length = win_length if win_length is not None \
            else self.n_fft
        self.window = window if window is not None \
            else torch.hann_window(self.win_length)
        self.center = center
        self.pad_mode = pad_mode
        self.normalized = normalized
        self.onesided = onesided if onesided is not None \
            else True
        time_differential_window = torch.fft.irfft(
            torch.fft.rfft(self.window, n=self.n_fft) * 1j*torch.arange(self.n_fft//2+1).float()
        )
        self.time_differential_window = time_differential_window.real
    
    def _stft_torch(self, x):
        return torch.stft(
            x,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            center=self.center,
            pad_mode=self.pad_mode,
            normalized=self.normalized,
            onesided=self.onesided,
            return_complex=True
        )
    
    def _diff_stft_torch(self, x):
        return torch.stft(
            x,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.time_differential_window,
            center=self.center,
            pad_mode=self.pad_mode,
            normalized=self.normalized,
            onesided=self.onesided,
            return_complex=True
        )


    def _istft(self, x, length=None):
        x = x.contiguous()
        return torch.istft(
            x,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            center=self.center,
            normalized=self.normalized,
            onesided=self.onesided,
            length=length,
            return_complex=False
        )

    def stft(self, x):
        """ 
        Args:
            x (torch.Tensor): shape (batch, time)

        Returns:
            ipc-effected X (torch.Tensor[torch.complex64]): shape (batch, freq, time)
            inv_operator (torch.Tensor[torch.complex64]): shape (batch, freq, time)
                : ipc-effected X * inv_operator = original X (= torch.stft(x))
        """
        X = self._stft_torch(x)
        M = torch.arange(X.shape[1], dtype=torch.int32, device=X.device).repeat_interleave(X.shape[2]).reshape(1, X.shape[1], X.shape[2])
        N = torch.arange(X.shape[2], dtype=torch.int32, device=X.device).repeat(X.shape[1]).reshape(1, X.shape[1], X.shape[2])
        spin_ = torch.exp(-2j*torch.pi*M*N*self.hop_length/self.n_fft)
        D = self._diff_stft_torch(x)
        X, D = X*spin_, D*spin_
        D_phi = -torch.imag(D/(X+1e-8))
        theta = torch.cumsum(D_phi, dim=2)
        theta = torch.exp(-2*torch.pi*1j*theta*self.hop_length/self.n_fft)
        X = X*theta
        inv_operator = torch.conj(theta*spin_)
        return X, inv_operator

    def istft(self, spectrogram, inv_D_spin=None):
        spectrogram = spectrogram * inv_D_spin if inv_D_spin is not None else spectrogram
        x = self._istft(spectrogram)
        return x

    

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np
    # Test
    stft = ipcSTFT(n_fft=2048, hop_length=512)
    stft_window = stft.window
    # watch window and time_differential_window
    plt.plot(np.arange(2048), stft_window.numpy(), label="window")
    plt.legend()
    plt.savefig("window.png")
    plt.close()
    plt.plot(np.arange(2048), stft.time_differential_window.numpy(), label="time_differential_window")
    plt.legend()
    plt.savefig("window_diff.png")
    plt.close()
    n = torch.arange(2048)
    w_d = (1/2)*torch.sin(2*np.pi*n/(2048-1))
    plt.plot(np.arange(2048), w_d.numpy(), label="window")
    plt.legend()
    plt.savefig("window_calculation.png")
    wave, _ = torchaudio.load("./sample/JSUT_BASIC5000_0001.wav")
    wave, inv_D = stft.stft(wave)
    wave = stft.istft(wave, inv_D)