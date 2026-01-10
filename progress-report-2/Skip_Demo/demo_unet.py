import numpy as np
import soundfile as sf
import torch
import torch.nn as nn


class UNet(nn.Module):
    def __init__(
        self, input_channels=1, output_channels=1, features=[64, 128, 256, 512]
    ):
        super().__init__()
        self.encoder = nn.ModuleList()
        self.pool = nn.MaxPool2d(2, 2)
        in_ch = input_channels
        for f in features:
            self.encoder.append(self._block(in_ch, f))
            in_ch = f

        self.bottleneck = self._block(features[-1], features[-1] * 2)

        self.upconvs = nn.ModuleList()
        self.decoder = nn.ModuleList()
        rev = list(reversed(features))
        in_ch = features[-1] * 2
        for f in rev:
            self.upconvs.append(nn.ConvTranspose2d(in_ch, f, kernel_size=2, stride=2))
            self.decoder.append(self._block(f * 2, f))
            in_ch = f

        self.final = nn.Conv2d(features[0], output_channels, kernel_size=1)

    def _block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    # def forward(self, x):
    #     skips = []
    #     for enc in self.encoder:
    #         print(enc)
    #         x = enc(x)
    #         skips.append(x)
    #         x = self.pool(x)
    #
    #     x = self.bottleneck(x)
    #     skips = skips[::-1]
    #     for up, dec, skip in zip(self.upconvs, self.decoder, skips):
    #         x = up(x)
    #         if x.shape[-2:] != skip.shape[-2:]:
    #             x = nn.functional.interpolate(
    #                 x, size=skip.shape[-2:], mode="bilinear", align_corners=False
    #             )
    #         x = torch.cat([skip, x], dim=1)
    #         x = dec(x)
    #
    #     return self.final(x)
    def forward(self, x):
        print(f"\nInput: {x.shape}")
        skips = []

        # --- Encoder ---
        for i, enc in enumerate(self.encoder):
            x = enc(x)
            print(f"\nEncoder {i} out: {x.shape}")
            skips.append(x)
            x = self.pool(x)
            print(f"After pool {i}: {x.shape}")

        # --- Bottleneck ---
        x = self.bottleneck(x)
        print(f"\nBottleneck: {x.shape}")

        # --- Decoder ---
        skips = skips[::-1]
        for i, (up, dec, skip) in enumerate(zip(self.upconvs, self.decoder, skips)):
            x = up(x)
            print(f"\nUpconv {i}: {x.shape}, skip: {skip.shape}")
            if x.shape[-2:] != skip.shape[-2:]:
                x = nn.functional.interpolate(
                    x, size=skip.shape[-2:], mode="bilinear", align_corners=False
                )
                print(f"  → Interpolated to: {x.shape}")

            x = torch.cat([skip, x], dim=1)
            print(f"  → After concat: {x.shape}")
            x = dec(x)
            print(f"\nDecoder {i} out: {x.shape}")

        out = self.final(x)
        print(f"\nFinal output: {out.shape}")
        return out


def stft_hann(x, n_fft=2048, hop=512):
    win = torch.hann_window(n_fft, device=x.device, dtype=x.dtype)
    spec = torch.stft(
        x,
        n_fft=n_fft,
        hop_length=hop,
        win_length=n_fft,
        window=win,
        center=True,
        return_complex=True,
        normalized=True,
    )
    return spec


def istft_hann(spec, n_fft=2048, hop=512, length=None):
    win = torch.hann_window(n_fft, device=spec.device, dtype=spec.real.dtype)
    wav = torch.istft(
        spec,
        n_fft=n_fft,
        hop_length=hop,
        win_length=n_fft,
        window=win,
        center=True,
        length=length,
    )
    return wav


class VocalSeparator:
    def __init__(self, model_path=None, device=None, n_fft=2048, hop=512):
        self.device = (
            torch.device(device)
            if device
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.n_fft = n_fft
        self.hop = hop

        # Build a default model first
        self.model = UNet(1, 1).to(self.device)

        if model_path:
            ckpt = torch.load(model_path, map_location=self.device)
            state = (
                ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
            )
            keys = list(state.keys())

            def has_training_names():
                # Your checkpoint shows enc1/enc2/..., upconv4, dec4, out
                return any(k.startswith("enc1.") for k in keys) or "out.weight" in state

            if has_training_names():
                # ---- 1) Infer features from checkpoint (enc1..enc4 out-channels)
                feats = []
                for i in range(1, 5):
                    wkey = f"enc{i}.0.weight"
                    if wkey in state:
                        feats.append(state[wkey].shape[0])
                if not feats:
                    raise RuntimeError("Could not infer features from checkpoint.")
                # Rebuild model with the inferred channel sizes
                self.model = UNet(1, 1, features=feats).to(self.device)

                # ---- 2) Remap state_dict keys -> demo model naming
                remapped = {}
                # enc1..enc4  → encoder.0..encoder.3  (and keep submodule indices .0 .1 .3 .4)
                for i in range(1, 5):
                    for sub in ["0", "1", "3", "4"]:
                        for suf in [
                            "weight",
                            "bias",
                            "running_mean",
                            "running_var",
                            "num_batches_tracked",
                        ]:
                            skey = f"enc{i}.{sub}.{suf}"
                            dkey = f"encoder.{i - 1}.{sub}.{suf}"
                            if skey in state:
                                remapped[dkey] = state[skey]

                # bottleneck stays bottleneck if names match; otherwise try to map from "bottleneck.*" directly
                for sub in ["0", "1", "3", "4"]:
                    for suf in [
                        "weight",
                        "bias",
                        "running_mean",
                        "running_var",
                        "num_batches_tracked",
                    ]:
                        skey = f"bottleneck.{sub}.{suf}"
                        if skey in state:
                            remapped[skey] = state[skey]

                # upconv4..1 → upconvs.0..3
                for i, idx in zip([4, 3, 2, 1], [0, 1, 2, 3]):
                    skey_w, skey_b = f"upconv{i}.weight", f"upconv{i}.bias"
                    dkey_w, dkey_b = f"upconvs.{idx}.weight", f"upconvs.{idx}.bias"
                    if skey_w in state:
                        remapped[dkey_w] = state[skey_w]
                    if skey_b in state:
                        remapped[dkey_b] = state[skey_b]

                # dec4..1 → decoder.0..3 with same (.0 .1 .3 .4) structure
                for i, idx in zip([4, 3, 2, 1], [0, 1, 2, 3]):
                    for sub in ["0", "1", "3", "4"]:
                        for suf in [
                            "weight",
                            "bias",
                            "running_mean",
                            "running_var",
                            "num_batches_tracked",
                        ]:
                            skey = f"dec{i}.{sub}.{suf}"
                            dkey = f"decoder.{idx}.{sub}.{suf}"
                            if skey in state:
                                remapped[dkey] = state[skey]

                # out → final
                if "out.weight" in state:
                    remapped["final.weight"] = state["out.weight"]
                if "out.bias" in state:
                    remapped["final.bias"] = state["out.bias"]

                # Load with strict=False to tolerate tiny BN bookkeeping diffs
                missing, unexpected = self.model.load_state_dict(remapped, strict=False)
                if missing or unexpected:
                    print(
                        "Loaded with non-strict mapping. Missing:",
                        missing,
                        " Unexpected:",
                        unexpected,
                    )
            else:
                # Plain state_dict shape; just load directly
                self.model.load_state_dict(state)

        self.model.eval()

    @torch.no_grad()
    def separate_vocals(self, audio_path, out_vocals="vocals.wav", out_accomp=None):
        # Load with soundfile
        wav_np, sr = sf.read(audio_path, always_2d=True)  # (samples, ch)
        if wav_np.shape[1] > 1:
            wav_np = wav_np.mean(axis=1, keepdims=True)
        x = torch.from_numpy(wav_np.T).float().to(self.device)  # (1, T)
        x = x.squeeze(0)  # (T) for torch.stft

        # STFT
        spec = stft_hann(x, n_fft=self.n_fft, hop=self.hop)
        mag = spec.abs()
        phase = torch.angle(spec)

        # Normalize
        mean = mag.mean()
        std = mag.std() + 1e-8
        mag_norm = (mag - mean) / std

        # Model expects (B, 1, F, frames)
        inp = mag_norm.unsqueeze(0).unsqueeze(0)
        mask = torch.sigmoid(self.model(inp)).squeeze(0).squeeze(0)

        # Apply mask and invert
        est_mag = mask * mag
        est_spec = est_mag * torch.exp(1j * phase)
        y = istft_hann(est_spec, n_fft=self.n_fft, hop=self.hop, length=x.shape[-1])

        # Save vocals (mono) with soundfile
        y_np = y.detach().cpu().numpy().astype(np.float32)  # shape: (T,)
        sf.write(out_vocals, y_np, sr)

        # Optionally save accompaniment = mix - vocals
        if out_accomp:
            mix_np = x.detach().cpu().numpy().astype(np.float32)  # shape: (T,)
            acc_np = np.clip(mix_np - y_np, -1.0, 1.0)
            sf.write(out_accomp, acc_np, sr)

        return y.detach().cpu(), sr


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("--out_vocals", default="vocals.wav")
    ap.add_argument("--out_accomp", default=None)
    ap.add_argument("--model", default=None)
    args = ap.parse_args()

    sep = VocalSeparator(model_path=args.model)
    sep.separate_vocals(args.input, args.out_vocals, args.out_accomp)
