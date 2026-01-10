# PyTorch U-Net Tutorial

Learn how U-Net works by training it on image segmentation, then apply it to audio stem separation.

**Learning Path:**
1. **Train on images** (this tutorial) → Understand U-Net basics
2. **Apply to audio** (`vocal_separation_sanity_check/`) → Your stem separation project

---

## 📋 Prerequisites

**System Requirements:**
- Python 3.8+
- ~2GB disk space for checkpoints

**Install Dependencies:**
```bash
uv pip install torch torchvision pillow numpy matplotlib tqdm wandb
```

---

## 🧠 What is U-Net?

U-Net is a neural network architecture designed for **image segmentation** - predicting a label for every pixel.

**How it works:**
1. **Encoder (Downsampling):** Compress image → extract features
2. **Decoder (Upsampling):** Expand features → reconstruct at original resolution
3. **Skip Connections:** Copy fine details from encoder to decoder

**For this tutorial:**
- **Input:** RGB image of a car (3 channels, 256×256)
- **Output:** Binary mask showing car pixels (1 channel, 256×256)

**Why this matters for audio:**
Audio spectrograms are 2D images. U-Net can learn to separate vocals from background by treating it as an image segmentation problem.

---

## ✓ Step 0: Test PyTorch Installation

Before training, verify PyTorch is installed and working:

**Mac/Linux:**
```bash
python3 test_pytorch.py
```

**Windows:**
```powershell
python test_pytorch.py
```

**Expected output:**
```
============================================================
PyTorch Installation Test
============================================================
✓ PyTorch imported successfully
  Version: 2.x.x

============================================================
Hardware Acceleration
============================================================
CUDA (NVIDIA GPU): ✗ Not available
MPS (Apple Silicon): ✓ Available
  Running on Apple Silicon GPU

============================================================
Tensor Operations Test
============================================================
x = [1.0, 2.0, 3.0]
y = [4.0, 5.0, 6.0]
x + y = [5.0, 7.0, 9.0]
Expected: [5.0, 7.0, 9.0]
✓ Tensor math works correctly

============================================================
Matrix Operations Test
============================================================
A @ B =
tensor([[ 4.,  6.],
        [10., 12.]])
Expected:
  [[4.0, 6.0],
   [10.0, 12.0]]
✓ Matrix multiplication works correctly

============================================================
Summary
============================================================
✓ All tests passed!
✓ PyTorch is ready to use
```

**What this tests:**
- ✅ PyTorch imports correctly
- ✅ Hardware acceleration detection (CUDA/MPS/CPU)
- ✅ Tensor addition: `[1,2,3] + [4,5,6] = [5,7,9]`
- ✅ Matrix multiplication works

**If you see errors:**
```bash
uv pip install --upgrade torch torchvision
```

### Try It Yourself

Now that PyTorch works, try creating tensors manually in the Python CLI:

**Mac/Linux:**
```bash
python3
```

**Windows:**
```powershell
python
```

**Then type these commands one by one:**
```python
>>> import torch
>>> x = torch.tensor([10.0, 20.0, 30.0])
>>> print(x)
tensor([10., 20., 30.])

>>> y = x * 2
>>> print(y)
tensor([20., 40., 60.])

>>> matrix = torch.randn(3, 3)
>>> print(matrix)
tensor([[ 0.1234, -0.5678,  0.9012],
        [ 1.2345, -0.6789,  0.3456],
        [-0.7890,  0.4567,  0.1234]])

>>> print(matrix.shape)
torch.Size([3, 3])

>>> exit()
```

**What you just did:**
- Created a 1D tensor (vector) with 3 numbers
- Multiplied every element by 2
- Created a random 3×3 matrix
- Checked its shape

This is what U-Net does internally - creates tensors, does math on them, reshapes them through layers.

### Test U-Net Architecture

Now test the actual U-Net model with random data:

**Mac/Linux:**
```bash
python3 -c "
from unet import UNet
import torch

print('📥 INPUT:')
x = torch.randn(1, 3, 572, 572)
print(f'   Shape: {x.shape}')
print(f'   Min: {x.min():.3f}, Max: {x.max():.3f}')

print('\n⚙️  Processing through U-Net...')
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
model = UNet(n_channels=3, n_classes=2).to(device)
model.eval()

with torch.no_grad():
    output = model(x.to(device))

print('\n📤 OUTPUT:')
print(f'   Shape: {output.shape}')
print(f'   Min: {output.min():.3f}, Max: {output.max():.3f}')
print(f'   Device: {output.device}')
print(f'\n✅ U-Net working on {device}!')
"
```

**Windows:**
```powershell
python -c "from unet import UNet; import torch; print('📥 INPUT:'); x = torch.randn(1, 3, 572, 572); print(f'   Shape: {x.shape}'); print(f'   Min: {x.min():.3f}, Max: {x.max():.3f}'); print('\n⚙️  Processing...'); device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'); model = UNet(n_channels=3, n_classes=2).to(device); model.eval(); output = model(x.to(device)); print('\n📤 OUTPUT:'); print(f'   Shape: {output.shape}'); print(f'   Min: {output.min():.3f}, Max: {output.max():.3f}'); print(f'\n✅ U-Net working!')"
```

**What you'll see:**
- Input: Random 3-channel image (572×572)
- Output: 2-channel prediction (same size)
- U-Net transforms input → output using learned patterns

### Test Training Pipeline

Test the forward and backward pass (what happens during training):

**Mac/Linux:**
```bash
python3 -c "
import torch
from unet import UNet
from PIL import Image
import numpy as np

print('📂 Loading test data...')
img = np.array(Image.open('data/imgs/test_0.png'))
mask = np.array(Image.open('data/masks/test_0_mask.png'))

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
model = UNet(n_channels=3, n_classes=2).to(device)

img_tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0
mask_tensor = torch.from_numpy(mask).unsqueeze(0).long().to(device) // 255

print('⚡ Forward pass...')
output = model(img_tensor)

print('📊 Computing loss...')
criterion = torch.nn.CrossEntropyLoss()
loss = criterion(output, mask_tensor)
print(f'   Loss: {loss.item():.4f}')

print('⬅️  Backward pass...')
loss.backward()

print('✅ Training pipeline fully functional!')
"
```

**Windows:**
```powershell
python -c "import torch; from unet import UNet; from PIL import Image; import numpy as np; print('📂 Loading...'); img = np.array(Image.open('data/imgs/test_0.png')); mask = np.array(Image.open('data/masks/test_0_mask.png')); device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'); model = UNet(n_channels=3, n_classes=2).to(device); img_t = torch.from_numpy(img).permute(2,0,1).unsqueeze(0).float().to(device)/255.0; mask_t = torch.from_numpy(mask).unsqueeze(0).long().to(device)//255; print('⚡ Forward...'); output = model(img_t); criterion = torch.nn.CrossEntropyLoss(); loss = criterion(output, mask_t); print(f'Loss: {loss.item():.4f}'); loss.backward(); print('✅ Pipeline works!')"
```

**What this shows:**
- Loads real car image + mask
- Forward pass: image → U-Net → prediction
- Loss: How wrong the prediction is
- Backward pass: Calculate gradients to improve

### Mini Training Loop

Run a tiny training session to see the complete cycle:

**Mac/Linux:**
```bash
python3 -c "
import torch
import numpy as np
from unet import UNet

print('🔧 Setup...')
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
model = UNet(n_channels=3, n_classes=2).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = torch.nn.CrossEntropyLoss()
print(f'   Device: {device}')

print('\n📊 Creating fake training data...')
x = torch.rand(1, 3, 256, 256).to(device)
y = torch.randint(0, 2, (1, 256, 256)).to(device)

print('\n🏋️  Training for 10 epochs...')
for epoch in range(10):
    model.train()
    optimizer.zero_grad()

    output = model(x)
    loss = criterion(output, y)
    loss.backward()
    optimizer.step()

    print(f'   Epoch {epoch+1}/10 - Loss: {loss.item():.4f}')

print('\n💾 Saving trained model...')
torch.save(model.state_dict(), 'mini_unet.pth')
print('✅ Model saved as mini_unet.pth')
print('   (This is fake training - use train.py for real training)')
"
```

**Windows:**
```powershell
python -c "import torch; from unet import UNet; print('🔧 Setup...'); device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'); model = UNet(n_channels=3, n_classes=2).to(device); optimizer = torch.optim.Adam(model.parameters(), lr=1e-4); criterion = torch.nn.CrossEntropyLoss(); x = torch.rand(1,3,256,256).to(device); y = torch.randint(0,2,(1,256,256)).to(device); print('\n🏋️  Training...'); [print(f'Epoch {e+1}/10 - Loss: {(model.train(), optimizer.zero_grad(), loss := criterion(model(x), y), loss.backward(), optimizer.step(), loss.item())[5]:.4f}') for e in range(10)]; torch.save(model.state_dict(), 'mini_unet.pth'); print('\n✅ Saved as mini_unet.pth')"
```

**What you learned:**
- Complete training cycle: forward → loss → backward → update
- Loss decreases as model learns
- Saved `.pth` file contains learned weights

---

## 🚀 Step 1: Train the Model (For Real)

Train U-Net to segment cars from images:

**Mac/Linux:**
```bash
python3 train.py --epochs 10 --batch-size 2 --validation 20
```

**Windows:**
```powershell
python train.py --epochs 10 --batch-size 2 --validation 20
```

**What happens:**
- Loads 5 car images from `data/imgs/`
- Loads ground truth masks from `data/masks/`
- Trains for 10 epochs (10 passes through the dataset)
- Saves model checkpoints to `checkpoints/`

**Expected output:**
```
INFO: Creating dataset with 5 examples
Epoch 1/10: Loss = 1.23
Epoch 2/10: Loss = 1.18
Epoch 3/10: Loss = 0.95
...
Epoch 10/10: Loss = 0.60
INFO: Checkpoint 10 saved!
```

**What's happening:**
- Each epoch, the model predicts car masks
- **Loss** measures prediction error (lower = better)
- Model adjusts weights to minimize loss
- After 10 epochs, trained weights saved to `checkpoints/checkpoint_epoch10.pth`

---

## 🔮 Step 2: Make Predictions

Use your trained model to segment a new image:

**Mac/Linux:**
```bash
python3 predict.py -m checkpoints/checkpoint_epoch10.pth -i data/imgs/test_0.png -o prediction.png
```

**Windows:**
```powershell
python predict.py -m checkpoints/checkpoint_epoch10.pth -i data/imgs/test_0.png -o prediction.png
```

**What happens:**
- Loads trained model weights
- Runs U-Net on `test_0.png`
- Saves predicted mask to `prediction.png`

**Expected output:**
```
INFO: Loading model checkpoints/checkpoint_epoch10.pth
INFO: Predicting image data/imgs/test_0.png
INFO: Mask saved to prediction.png
```

**Now view your predicted mask:**
```bash
open prediction.png    # Mac
start prediction.png   # Windows
```

You should see a **mostly black/white image** where:
- **White/light gray pixels** = car (foreground)
- **Black/dark gray pixels** = background

**What to expect:**
The mask will likely look **noisy or textured** (like fabric or static) rather than perfectly clean. This is normal because:

1. **Only 5 training images** - The model hasn't seen enough data to generalize perfectly
2. **Only 10 epochs** - Limited training time
3. **Pixel-level uncertainty** - The model assigns confidence values to each pixel, creating gradients between 0-255

**Good segmentation:** White blob roughly matches the car shape, even if edges are fuzzy or speckled
**Bad segmentation:** Random white dots everywhere, or entirely black/white image

Compare it to the original `data/imgs/test_0.png` - does the general shape match?

**Why this matters for audio:**
Audio spectrograms will have the same "noisy" quality. The model separates vocals from background, but won't be perfect - you'll hear some artifacts. More training data and epochs improve quality.

**You just completed the image segmentation tutorial!**

---

## ✅ Step 3: Verify Prediction Numbers (Optional)

If you want numerical verification, check the prediction stats:

**Mac/Linux:**
```bash
python3 -c "
from PIL import Image
import numpy as np
pred = np.array(Image.open('prediction.png'))
print(f'Shape: {pred.shape}')
print(f'Value range: [{pred.min()}, {pred.max()}]')
print(f'Foreground pixels: {(pred > 0).mean() * 100:.1f}%')
"
```

**Windows (PowerShell):**
```powershell
python -c "from PIL import Image; import numpy as np; pred = np.array(Image.open('prediction.png')); print(f'Shape: {pred.shape}'); print(f'Foreground: {(pred > 0).mean() * 100:.1f}%')"
```

**Expected output:**
```
Shape: (256, 256)
Value range: [0, 255]
Foreground pixels: 22.3%
```

**What to look for:**
- Foreground should be **15-35%** for car segmentation
- If <5%: Model barely found anything (undertrained)
- If >60%: Model over-segmented (too aggressive)

---

## 🔧 Training Options

### Adjust Epochs

**Mac/Linux:**
```bash
python3 train.py --epochs 5   # Quick test (less accurate)
python3 train.py --epochs 50  # Longer training (more accurate)
```

**Windows:**
```powershell
python train.py --epochs 5   # Quick test (less accurate)
python train.py --epochs 50  # Longer training (more accurate)
```

**What epochs control:**
- More epochs = model sees data more times
- More epochs = better learning (up to a point)
- Too many epochs = overfitting (model memorizes instead of learns)

### Adjust Batch Size

**Mac/Linux:**
```bash
python3 train.py --batch-size 1  # Process 1 image at a time
python3 train.py --batch-size 4  # Process 4 images at a time
```

**Windows:**
```powershell
python train.py --batch-size 1  # Process 1 image at a time
python train.py --batch-size 4  # Process 4 images at a time
```

**What batch size controls:**
- Larger batches = faster training (parallel GPU processing)
- Larger batches = more stable gradients but less frequent updates
- Smaller batches = slower but more frequent weight updates

**Recommendation:** Start with `--batch-size 2`, adjust based on your GPU memory.

---

## 🎓 What You Learned

### Core Concepts

**Tensors:**
Multi-dimensional arrays that PyTorch uses to store data:
- Image: `(3, 256, 256)` = 3 color channels, 256×256 pixels
- Batch of images: `(8, 3, 256, 256)` = 8 images at once

**U-Net Architecture:**
- **Encoder:** Downsample image → extract features (what's in the image?)
- **Decoder:** Upsample features → predict mask (where is it?)
- **Skip connections:** Preserve spatial details lost during downsampling

**Training Loop:**
1. **Forward pass:** Run images through U-Net → get predictions
2. **Calculate loss:** Compare predictions to ground truth
3. **Backpropagation:** Calculate gradients (how to improve)
4. **Update weights:** Adjust model parameters to reduce loss

**Checkpoints (`.pth` files):**
Saved model weights. Contains the learned patterns, not the architecture.

---

## ✨ Next: Apply to Audio Stem Separation

You now understand:
- ✅ How U-Net processes 2D data (images)
- ✅ How training adjusts weights to minimize error
- ✅ How to make predictions with a trained model

**Ready for the real project:**

```bash
cd vocal_separation_sanity_check
cat README.md
```

**What changes for audio:**
- **Input:** Spectrogram of full song (2D image-like representation of audio)
- **Output:** Spectrogram of isolated vocals
- **Same U-Net architecture, different data**

The concepts are identical:
- PNG images → Audio spectrograms (created with **librosa**)
- Car pixel masks → Vocal frequency masks
- RGB channels (3) → Frequency bins (hundreds)
- `.pth` checkpoints → Same format

**New concept: Librosa**
You'll learn how **librosa** converts `.wav` audio files → spectrograms (2D arrays) that U-Net can process. Then converts spectrograms back → `.wav` audio files.

**The pipeline:**
1. Audio `.wav` → Spectrogram (librosa)
2. Spectrogram → U-Net → Vocal mask
3. Apply mask → Isolated vocal spectrogram
4. Spectrogram → Audio `.wav` (librosa)

Read `vocal_separation_sanity_check/README.md` to see how audio processing works.

---

## 🐛 Troubleshooting

**"No module named 'torch'"**
```bash
uv pip install torch torchvision
```

**"CUDA out of memory"**
Reduce batch size:
```bash
# Mac/Linux
python3 train.py --batch-size 1

# Windows
python train.py --batch-size 1
```

**"FileNotFoundError: data/imgs/"**
Make sure you're in the project root directory:
```bash
cd /path/to/Pytorch-UNet
ls data/imgs/  # Should show image files
```

**Training loss not decreasing**
- Try more epochs: `python3 train.py --epochs 50` (Mac/Linux) or `python train.py --epochs 50` (Windows)
- Check learning rate in `train.py` (should be ~0.001)
- Verify data is loading correctly

**Prediction is all black**
- Model needs more training epochs
- Check that checkpoint file exists and loaded correctly
- Verify input image is valid
