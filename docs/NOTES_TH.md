# NOTES: บันทึกผล Phase 0–3

> ไฟล์นี้คือ "สมุดแล็บ" ของโปรเจกต์ ทุกตัวเลขในนี้มาจากการรันจริง
> สคริปต์และ output ดิบอยู่ในโฟลเดอร์ `phase0_1/`

**รันเมื่อ:** 15 ต.ค. 2026 บน cloud CPU (2 cores, torch 2.14.1, Python 3.13) โดยใช้ checkpoint และข้อมูลจากเครื่องคุณ

**ยังต้องรันบนเครื่อง Windows ของคุณเอง** (ดูหัวข้อ 0.1 / 0.2 / 0.4-GPU ด้านล่าง) ผมเข้าถึง Python และ GPU บน Windows ของคุณจากตรงนี้ไม่ได้

---

## Phase 0: Environment และ MedSAM

### 0.1 Hardware ✅

ให้รันคำสั่งนี้ในโฟลเดอร์ `prescreening` ใน env ที่ใช้รัน repo:

```
python check_env.py > phase0_1\env_windows.txt
```

แล้วกรอกตารางนี้:

| รายการ | ค่า |
|---|---|
| GPU |NVIDIA GeForce RTX 3050 Ti Laptop GPU |
| VRAM |4.0 GB |
| torch / CUDA | 2.2.1+cu121|
| Python |3.11.17 |
| OS | Windows |

### 0.2 Environment freeze ✅

**สรุป:** package หลักตรงกับ `requirements.txt` ทุกตัว ได้แก่ torch 2.2.1+cu121 (เวอร์ชันเดียวกับที่ README ของ paper ใช้), lightning 2.4.0, torchmetrics 1.4.3, geomstats 2.7.0, medmnist 3.0.1 และ numpy 1.25.2 รายการเต็มอยู่ใน `phase0_1/env_freeze_windows.txt`

```
pip freeze > phase0_1\env_freeze_windows.txt
```

เทียบกับ `2026-isbi-mayr/requirements.txt` โดยดู torch, lightning, geomstats และ medmnist เป็นหลัก ถ้าเวอร์ชันไม่ตรง ให้จดไว้ตรงนี้

From `2026-isbi-mayr/requirements.txt`
GitPython==3.1.43
Jinja2==3.1.4
MarkupSafe==2.1.2
PyYAML==6.0
SimpleITK==2.3.1
aiohappyeyeballs==2.4.3
aiohttp==3.10.10
aiosignal==1.3.1
attrs==23.1.0
certifi==2022.12.7
charset-normalizer==2.1.1
click==8.1.3
colorama==0.4.6
contourpy==1.0.7
cycler==0.11.0
docker-pycreds==0.4.0
filelock==3.9.0
fire==0.5.0
fonttools==4.39.4
frozenlist==1.4.1
fsspec==2024.2.0
geomstats==2.7.0
gitdb==4.0.11
idna==3.4
imageio==2.28.1
ipympl==0.9.4
ipywidgets==8.0.6
joblib==1.2.0
jupyterlab==4.3.0
kiwisolver==1.4.4
lazy_loader==0.2
lightning-utilities==0.11.8
lightning==2.4.0
matplotlib==3.7.1
medmnist==3.0.1
monai==1.4.0
mpmath==1.2.1
multidict==6.1.0
networkx==3.2.1
nibabel==5.2.1
numpy==1.25.2
opencv-python==4.10.0.84
packaging==23.1
pandas==2.0.1
pillow==10.3.0
platformdirs==3.5.1
propcache==0.2.0
protobuf==4.25.3
psutil==5.9.5
pyparsing==3.0.9
python-dateutil==2.8.2
pytorch-lightning==2.4.0
pytz==2023.3
requests==2.32.3
scikit-image==0.20.0
scikit-learn==1.5.1
scipy==1.9.3
sentry-sdk==2.17.0
setproctitle==1.3.3
six==1.16.0
smmap==5.0.1
sympy==1.11.1
termcolor==2.4.0
threadpoolctl==3.1.0
tifffile==2023.4.12
torchmetrics==1.4.3
tqdm==4.66.6
typing_extensions==4.8.0
tzdata==2023.3
urllib3==1.26.13
yarl==1.15.4
wheel==0.38.4

From phase0_1\env_freeze_windows.txt
absl-py==2.5.0
aiohappyeyeballs==2.4.3
aiohttp==3.10.10
aiosignal==1.3.1
annotated-types==0.8.0
anyio==4.14.2
argon2-cffi==25.1.0
argon2-cffi-bindings==26.1.0
arrow==1.4.0
asttokens==3.0.2
async-lru==2.3.0
attrs==23.1.0
babel==2.18.0
beautifulsoup4==4.15.0
bleach==6.4.0
certifi==2022.12.7
cffi==2.1.1
charset-normalizer==2.1.1
click==8.5.0
colorama==0.4.6
comm==0.2.3
contourpy==1.0.7
cycler==0.11.0
debugpy==1.8.22
decorator==5.3.1
defusedxml==0.7.1
docker-pycreds==0.4.0
executing==2.2.1
fastjsonschema==2.22.2
filelock==3.9.0
fire==0.5.0
fonttools==4.39.4
fqdn==1.6.0
frozenlist==1.4.1
fsspec==2024.2.0
geomstats==2.7.0
gitdb==4.0.11
GitPython==3.1.43
googleapis-common-protos==1.75.5
grpcio==1.84.0
h11==0.16.0
httpcore==1.0.9
httpx==0.28.1
idna==3.4
imageio==2.28.1
ipykernel==7.2.0
ipympl==0.9.4
ipython==8.39.0
ipython-genutils==0.2.0
ipywidgets==8.0.6
isoduration==20.11.0
jedi==0.20.0
Jinja2==3.1.4
joblib==1.2.0
json5==0.15.0
jsonpointer==3.1.1
jsonschema==4.26.0
jsonschema-specifications==2025.9.1
jupyter-events==0.12.1
jupyter-lsp==2.3.1
jupyter_client==8.8.0
jupyter_core==5.9.1
jupyter_server==2.21.1
jupyter_server_terminals==0.5.4
jupyterlab==4.3.0
jupyterlab_pygments==0.3.0
jupyterlab_server==2.28.1
jupyterlab_widgets==3.0.17
kiwisolver==1.4.4
lark==1.3.1
lazy_loader==0.2
lightning==2.4.0
lightning-utilities==0.11.8
Markdown==3.11
MarkupSafe==2.1.2
matplotlib==3.7.1
matplotlib-inline==0.2.2
medmnist==3.0.1
-e git+https://github.com/bowang-lab/MedSAM@d71e8a1a99ad751840a22a7fa3ecfb4166fb1488#egg=medsam
mistune==3.3.4
monai==1.4.0
mpmath==1.2.1
multidict==6.1.0
nbclient==0.11.0
nbconvert==7.17.1
nbformat==5.11.1
nest-asyncio==1.6.0
networkx==3.2.1
nibabel==5.2.1
notebook_shim==0.2.4
numpy==1.25.2
opencv-python==4.10.0.84
opentelemetry-api==1.45.1
opentelemetry-exporter-http-transport==0.66b1
opentelemetry-exporter-otlp-common==0.66b1
opentelemetry-exporter-otlp-proto-common==1.45.1
opentelemetry-exporter-otlp-proto-http==1.45.1
opentelemetry-proto==1.45.1
opentelemetry-sdk==1.45.1
opentelemetry-semantic-conventions==0.66b1
overrides==7.7.0
packaging==23.1
pandas==2.0.1
pandocfilters==1.5.1
parso==0.8.7
pillow==10.3.0
platformdirs==3.5.1
prometheus_client==0.26.0
prompt_toolkit==3.0.53
propcache==0.2.0
protobuf==7.36.2
psutil==5.9.5
pure_eval==0.2.4
pycparser==3.0
pydantic==2.13.5
pydantic_core==2.46.5
Pygments==2.21.0
pyparsing==3.0.9
python-dateutil==2.8.2
python-json-logger==4.2.0
pytorch-lightning==2.4.0
pytz==2023.3
PyWavelets==1.9.0
pywinpty==3.0.5
PyYAML==6.0
pyzmq==27.2.0
referencing==0.37.0
requests==2.32.3
rfc3339-validator==0.1.4
rfc3986-validator==0.1.1
rfc3987-syntax==1.1.0
rpds-py==2026.9.1
scikit-image==0.20.0
scikit-learn==1.5.1
scipy==1.9.3
Send2Trash==2.1.0
sentry-sdk==2.17.0
setproctitle==1.3.3
SimpleITK==2.3.1
six==1.16.0
smmap==5.0.1
soupsieve==2.10
stack-data==0.6.3
sympy==1.11.1
tensorboard==2.21.0
tensorboard-data-server==0.7.2
termcolor==2.4.0
terminado==0.18.1
threadpoolctl==3.1.0
tifffile==2023.4.12
tinycss2==1.5.1
torch==2.2.1+cu121
torchaudio==2.2.1+cu121
torchmetrics==1.4.3
torchvision==0.17.1+cu121
tornado==6.5.10
tqdm==4.66.6
traitlets==5.16.1
typing-inspection==0.4.4
typing_extensions==4.16.0
tzdata==2023.3
uri-template==1.3.0
urllib3==1.26.13
wcwidth==0.9.2
webcolors==25.10.0
webencodings==0.6.1
websocket-client==1.9.2
Werkzeug==3.1.9
widgetsnbextension==4.0.16
xxhash==4.0.1
yarl==1.15.4


### 0.3 MedSAM segmentation ✅

- รัน `MedSAM_Inference.py` (รูป demo CT ช่องท้อง พร้อม box) แล้ว
- ผลอยู่ที่ `phase0_1/phase0_medsam_segmentation_demo.png`
- mask สีเหลืองครอบก้อน (lesion) ในตับ ตรงกับ box ที่ให้ไป **สรุป: checkpoint และการติดตั้ง `segment_anything` ใช้งานได้ถูกต้อง**

**Checkpoint:** `models/medsam_vit_b.pth`, 375,049,145 bytes

```
SHA-256: 34b34b78c1d18cb8c6bf84cf9c00e135d6d6c965699f3c0e31ef1bc9dcb5be74
```

### 0.4 ด่าน shape ของ encoder ✅

รันผ่าน code path ของ official repo จริง (`common.gve.PretrainedFeatures`) ด้วย input 4 แบบ:

| Input | ขนาดเดิม | หลัง `transform_demo` | Output |
|---|---|---|---|
| random | (3, 64, 64) | (1, 3, 1024, 1024) | **(256, 64, 64)** |
| DermaMNIST train[0] | (3, 64, 64) | (1, 3, 1024, 1024) | **(256, 64, 64)** |
| Kvasir polyps | (3, 1064, 1350) | (1, 3, 1024, 1024) | **(256, 64, 64)** |
| Kvasir normal-z-line | (3, 1024, 1280) | (1, 3, 1024, 1024) | **(256, 64, 64)** |

ไม่ว่ารูปจะขนาดเท่าไหร่ ถูก resize เป็น 1024² ก่อนเสมอ จึงได้ feature ขนาดเท่ากันทุกรูป

**เวลาบน cloud CPU:** ประมาณ 16 วินาทีต่อรูป ถ้าใช้ CPU แบบนี้ DermaMNIST จะใช้ 45 ชั่วโมง และ Kvasir 36 ชั่วโมง **จึงต้องใช้ GPU** ให้รันคำสั่งนี้บน Windows เพื่อเอาตัวเลขจริง:

```
cd 2026-isbi-mayr\src
python ..\..\phase0_1\phase0_encoder_trace.py --model-dir ..\..\models --derma ..\..\data\medmnist\dermamnist\dermamnist_64.npz --kvasir ..\..\kvasir-dataset-v2\polyps\00072d5f-7cd8-434c-8a5a-1a0bb2c9711d.jpg
```

| รายการ | ค่าบน GPU |
|---|---|
| ms/รูป |431 ms |
| peak VRAM |2.74 GB |
| เวลาประมาณสำหรับ Derma |1.2 h |
| เวลาประมาณสำหรับ Kvasir |0.9 h |

### 0.5 Encoder แบบละเอียด (จาก forward hooks) ✅

```
input            (1, 3, 1024, 1024)
patch_embed      (1, 64, 64, 768)    Conv 16×16 stride 16 → 4096 patches, channels-last
+ pos_embed      (1, 64, 64, 768)    pos_embed มีขนาด (1, 64, 64, 768)
blocks[0..11]    (1, 64, 64, 768)    window_size = [14,14,0, 14,14,0, 14,14,0, 14,14,0]
                                     (0 = global attention ที่ block 2, 5, 8, 11)
permute          (1, 768, 64, 64)
neck[0] Conv1×1  (1, 256, 64, 64)    std ~20 (ยังไม่ normalize)
neck[1] LN2d     (1, 256, 64, 64)    std ~0.04
neck[2] Conv3×3  (1, 256, 64, 64)
neck[3] LN2d     (1, 256, 64, 64)    ← OUTPUT = image_embeddings = feature ที่ใช้
```

**ทำความเข้าใจ**
- ข้างใน ViT ขนาด tensor ไม่เปลี่ยนเลย คือ 64×64 patches × 768 ตลอดทั้ง 12 blocks
- สิ่งที่ blocks ทำคือผสมข้อมูลระหว่าง patch ผ่าน attention
- neck ทำหน้าที่ลดมิติ 768 → 256 แล้วจบด้วย LayerNorm
- ผลลัพธ์จึงมองได้ว่าเป็น **4096 ตัวอย่าง ของเวกเตอร์ 256 มิติ** (1 ตัวอย่างต่อ 1 patch ขนาด 16×16 pixel ของภาพ 1024²)

### 0.6 Parameters และการ freeze ✅

| ส่วน | Parameters |
|---|---|
| SAM ทั้งหมด | 93.74 M |
| image_encoder | **89.67 M (95.7%)** |
| prompt_encoder | 0.01 M |
| mask_decoder | 4.06 M |

- **ใช้เฉพาะ image_encoder** ส่วน prompt_encoder และ mask_decoder ไม่ถูกใช้เลย
- `requires_grad` ของ parameter ยังเป็น True อยู่ แต่ `PretrainedFeatures.__call__` เรียก encoder ภายใต้ `torch.no_grad()` และตั้ง `sam.eval()` ไว้แล้ว **MedSAM จึงถูก freeze จริง** ไม่มี gradient และไม่ถูกอัปเดต
- นี่คือเหตุผลที่ cache feature ได้: เมื่อ encoder ไม่เปลี่ยน feature ของแต่ละรูปก็เหมือนเดิมทุก epoch
- SAM เก็บ `pixel_mean/std` (ค่า ImageNet) ไว้ แต่ pipeline นี้ **ไม่ได้ใช้** เพราะเรียก `image_encoder` ตรง ๆ โดยข้าม `Sam.preprocess`

---

## Phase 1: ทำความเข้าใจโค้ด official

### วิธีอ่าน: ผมใส่ป้าย `📖` ไว้ในโค้ดแล้ว 69 จุด

- ใน VS Code กด **Ctrl+Shift+F** แล้วค้นหา `📖` จะเห็นทุกจุดเรียงตามไฟล์
- ถ้าต้องการเฉพาะเรื่อง ให้ค้นหา `📖 [P1.4` (หรือข้ออื่น)
- ป้ายที่มี ★ คือบรรทัดสำคัญที่สุด
- ป้ายเป็น comment อย่างเดียว ไม่ได้เปลี่ยนการทำงานของโค้ด (ตรวจแล้วว่าทุกไฟล์ compile ผ่าน)
- ถ้าต้องการลบป้ายออกทั้งหมด: `python phase0_1\add_read_markers.py --remove`
- ถ้าต้องการใส่กลับ: รันคำสั่งเดิมโดยไม่ใส่ `--remove`

**ลำดับการอ่าน (ข้อ 1.1)**

| ลำดับ | ไฟล์ | ค้นหาป้าย |
|---|---|---|
| 1 | `src/main.py` → `handle_spdnet_gve` | `[P1.1-①]` ถึง `[P1.1-④]` |
| 2 | `src/common/gve.py` → `PretrainedFeatures` | `[P1.1-⑤]`, `[P0.6]` |
| 3 | `src/pretrained/medsam.py` → `transform_demo`, `SamInvoker.embed` | `[P1.2]`, `[P1.1-⑥]` ★★ |
| 4 | `MedSAM/segment_anything/modeling/image_encoder.py` | `[P0.5]` |
| 5 | `src/common/__init__.py` → `cov_descr`, แล้วต่อที่ `src/torchutil/bcov.py` | `[P1.3]` |
| 6 | `src/experiments/__init__.py` → `prepare_datasets` และ `src/data.py` → `CachedDataset` | `[P1.1]`, `[P1.7-b]` |
| 7 | `src/experiments/spdnet.py` → `fit_spdnet` | `[P1.4]`, `[P1.5]`, `[P1.7]` |
| 8 | `src/nn_lit/spdnets.py`, `src/spdnet/functional.py`, `src/spdnet/optimizers.py` | `[P1.4]` |
| 9 | `src/nn_lit/__init__.py`, `src/layers/losses/__init__.py` | `[P1.5]` |

### 1.1 Pipeline สรุปเป็นภาษาคน

1. `handle_spdnet_gve` โหลด MedSAM ผ่าน `PretrainedFeatures`
2. สร้าง `mapper` ที่แปลงรูปหนึ่งรูปเป็น covariance ขนาด 256×256
3. `prepare_datasets` โหลด split official ของ DermaMNIST (7007/1003/2005) แล้วรัน mapper **ครั้งเดียวต่อรูป** เพื่อเขียน cache ลงไฟล์ `.nd`
4. `fit_spdnet` เทรน SPDNet ใหม่ 5 ครั้ง แล้วรายงาน**ค่าเฉลี่ย** test metrics

**ตัวอย่างการทำงาน:** รูป Derma 1 รูป (3,64,64) → ขยายเป็น 1024² → MedSAM → (256,64,64) → covariance (256,256) → SPDNet → logits 7 ค่า

### 1.2 Preprocessing: per-channel กับ global min-max ✅

| | วิธี min-max | ไฟล์ |
|---|---|---|
| 2026-isbi-mayr (official ที่เราจะ reproduce) | **แยกทีละ channel** | `torchutil/transforms.py`, `NormalizeMinMax(start_dim=2)` |
| MedSAM ต้นฉบับ | **ทั้งภาพ** (รวม 3 channels) | `MedSAM_Inference.py` บรรทัดประมาณ 121 |

**ผลการทดลอง (`phase1_checks.py`)**

| รูป | mean RGB แบบ per-channel | mean RGB แบบ global | cosine ของ features | covariance ต่างกัน |
|---|---|---|---|---|
| Derma | (0.66, 0.59, 0.61) | (0.83, 0.54, 0.55) | 0.875 | **74%** |
| Kvasir | (0.38, 0.24, 0.21) | (0.38, 0.24, 0.20) | 0.9997 | 2% |

**ทำความเข้าใจ**
- รูป Derma 64×64 มีช่วงค่าแคบ (0.31–0.83) และแต่ละ channel มีช่วงไม่เท่ากัน
- per-channel min-max ยืดแต่ละ channel ให้เต็ม [0,1] แยกกัน ผลคือ **สมดุลสีเปลี่ยน** สีแดงที่เด่นในรอยโรคผิวหนังจะถูกกดลงให้เท่ากับ G/B
- รูป Kvasir มีทั้ง pixel ดำ (ขอบ) และ pixel สว่างอยู่แล้วทุก channel ทั้งสองวิธีจึงแทบไม่ต่างกัน
- **ข้อสรุป:** ตอน reproduce ต้องใช้แบบ official (per-channel) ห้ามเปลี่ยน
- ความต่างนี้เป็นประเด็นที่ดีสำหรับ presentation: preprocessing มีผลต่อ feature ของ Derma มาก

### 1.3 Covariance ✅

- `cov_descr`: (256,64,64) → flatten → (256,4096) → ลบค่าเฉลี่ยของแต่ละ channel → `Z Zᵀ / (N+1)`
- ผลเทียบกับ `torch.cov` ซึ่งหารด้วย N−1: อัตราส่วนเท่ากับ **0.99951184 = 4095/4097 พอดี**
- **ข้อสรุป:** code หารด้วย N+1 ขณะที่ paper เขียนว่าหารด้วย N−1 เป็นการ scale คงที่ทั้ง matrix ไม่มีผลต่อการจำแนก แต่เป็นจุดที่ paper กับ code ไม่ตรงกัน
- ผลออกมาสมมาตรพอดี (max|C−Cᵀ| = 0)

**ทำไม covariance เป็น second-order และ GAP เป็น first-order**
- `feature_mean` ที่ bcov ลบทิ้งคือค่าเดียวกับ GAP ทุกประการ คือค่าเฉลี่ย 256 ค่าข้าม 4096 patches
- covariance จึง**ไม่มีข้อมูล first-order อยู่เลย** มันเก็บแค่ว่า channel i กับ j แปรผันไปด้วยกันอย่างไรข้ามตำแหน่งต่าง ๆ ในภาพ
- ส่วน GAP เก็บแค่ค่าเฉลี่ย และทิ้งความสัมพันธ์ระหว่าง channel ไป
- สองแบบนี้จึงเก็บข้อมูลคนละส่วนกันพอดี ซึ่งจะเป็นแกนของการวิเคราะห์ใน Phase 7

### 1.4 SPDNet ✅

ผลจริงจาก `BasicSPDNet(256, 7)` ของ official ป้อนด้วย covariance จริง batch ละ 4:

```
(4,256,256) → unsqueeze → (4,1,256,256)
bimap1 (4,1,128,128) → reeig → bimap2 (4,1,64,64) → reeig → bimap3 (4,1,32,32) → reeig
logeig (4,1,32,32) → flatten 1024 → Linear → (4,7)
```

| Parameter | Shape | ชนิด | จำนวน | Optimizer |
|---|---|---|---|---|
| bimap1.W | 256×128 | StiefelParameter | 32,768 | Riemannian SGD |
| bimap2.W | 128×64 | StiefelParameter | 8,192 | Riemannian SGD |
| bimap3.W | 64×32 | StiefelParameter | 2,048 | Riemannian SGD |
| linear | 7×1024 + 7 | Parameter | 7,175 | Adam |
| **รวม** | | | **50,183** | |

เทียบกับ GAP + Linear(256→8) ใน Phase 5 ซึ่งมีแค่ 2,056 parameters

**คำตอบข้อ 1.4**
- **ทำไม W ต้องอยู่บน Stiefel manifold (WᵀW = I)**
  - BiMap คำนวณ WᵀXW ถ้า X เป็น SPD และ W มี column ที่ orthonormal (full column rank) ผลลัพธ์ก็ยังเป็น SPD
  - การบังคับ WᵀW = I ยังช่วยกันไม่ให้ W ยุบ (degenerate) และกันไม่ให้ scale ระเบิด
  - เปรียบได้กับการ "หมุนแล้วตัดเลือก" subspace ขนาด 128 มิติจาก 256 มิติ
- **QR retraction ทำอะไร**
  - หลัง gradient step W จะหลุดออกจาก manifold
  - การทำ QR บน (W + step) ดึงกลับมาให้เป็น orthonormal
  - ผลจริง: ‖WᵀW − I‖ = 1e-5 ก่อน step และ **8e-15 หลัง step**
- **Adam อัปเดตอะไร:** เฉพาะ Linear (2 tensors) ส่วน W ทั้ง 3 ตัวใช้ Riemannian SGD ที่ไม่มี momentum
- **paper กับ code ไม่ตรงกัน:** paper เขียนว่า "manifold-adapted Adam"
- **ReEig:** ปรับ eigenvalue ที่ < 1e-4 ให้เป็น 1e-4 (ทำหน้าที่คล้าย ReLU แต่ทำบน eigenvalue) ใช้ module object ตัวเดียวกันทุกชั้น
- **LogEig:** คำนวณ U·log(S)·Uᵀ เพื่อพา matrix จาก manifold ไปยัง tangent space ซึ่งเป็น Euclidean ทำให้ใช้ Linear ได้อย่างมีความหมาย
- **ทำไมไม่ flatten covariance แล้วเข้า Linear ตรง ๆ**
  - matrix SPD อยู่บนกรวย (cone) ที่โค้ง ระยะทางแบบ Euclidean บนกรวยนี้บิดเบือน เช่น เกิด swelling effect
  - log ช่วยทำให้ space แบนลง
- **ต้นทุน:** ทุกชั้น ReEig และ LogEig ต้องทำ SVD ต่อทุกรูป (256→128→64→32) ส่วนนี้คือต้นทุนหลักของ SPDNet

### 1.5 Training wrapper ✅

| เรื่อง | ค่าใน code | ไฟล์ |
|---|---|---|
| Early stopping | `val/Balanced accuracy`, mode max, patience 15 | `experiments/spdnet.py` |
| Checkpoint | `val/loss`, mode min, save_top_k=1 ⚠ **คนละ metric กับ ES** | `nn_lit/__init__.model_checkpoint` |
| Weights ที่นำไป test | checkpoint ที่ val loss ต่ำสุด (`load_from_checkpoint` หลัง fit) | `nn_lit/__init__.BaseModel.fit` |
| Loss | cross-entropy × w[target] แล้ว mean | `layers/losses` |
| Class weight | w_c = (1/n_c) / Σ_k (1/n_k) | `layers/losses` |
| Batch | 32 สำหรับ train (shuffle), 256 สำหรับ val/test | `BaseModel._dataloader` |
| LR / scheduler | 1e-3, ไม่มี scheduler | `fit_spdnet`, `configure_optimizers` |
| Epochs / repeats | สูงสุด 250 / 5 รอบ, รายงานค่าเฉลี่ย | `fit_spdnet` |
| "Balanced accuracy" | macro-averaged accuracy = ค่าเฉลี่ย recall ต่อ class | `ClassifierModel._create_metrics` |

**DermaMNIST (นับจากไฟล์จริง)**

| class | 0 | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|---|
| จำนวนใน train | 228 | 359 | 769 | 80 | 779 | **4693** | 99 |
| loss weight | 0.135 | 0.086 | 0.040 | **0.384** | 0.039 | **0.0065** | 0.310 |

- ถ้าโมเดลทายเป็น class 5 (nevus) ทุกรูป จะได้ accuracy 0.670 แต่ balanced accuracy แค่ 0.143
- **นี่คือเหตุผลที่ paper รายงาน balanced accuracy** และใช้ class weight
- เป้าหมายจาก paper (BA 56.3) จึงดีกว่าการทายสุ่มตามสัดส่วนอย่างชัดเจน

### 1.6 SPD หรือ PSD? ผลยืนยันสมมติฐาน ✅

**สมมติฐาน:** neck ของ encoder จบด้วย LayerNorm2d ซึ่งที่ทุก pixel ทำให้ Σ_c (y_c − β_c)/γ_c = 0 ดังนั้นเวกเตอร์ v = 1/γ ควรเป็น null vector ของ covariance

| | Derma train[0] | Kvasir polyps |
|---|---|---|
| eigenvalue เล็กสุด (float64) | **8.2e-19** | **7.1e-19** |
| λ₁ / λ₂ | 1.8e-10 | 2.5e-10 |
| numerical rank | **255 / 256** | **255 / 256** |
| \|cos(eigvec_min, 1/γ)\| | **1.000000** | **1.000000** |
| ‖Cv‖ / ‖C‖ | 2.9e-11 | 4.0e-11 |
| จำนวน eigenvalue ที่ < 1e-4 (threshold ของ ReEig) | 59 | 69 |
| eigenvalue เล็กสุดใน float32 (แบบที่ cache เก็บ) | 4.0e-11 | **−1.8e-11** |

**ข้อสรุป (ยืนยันแล้ว)**
- covariance ของ MedSAM ทุกรูป**มี rank 255 ไม่ใช่ 256** จึงเป็น PSD ไม่ใช่ SPD อย่างแท้จริง
- null direction ตรงกับ 1/γ ของ LayerNorm สุดท้ายพอดี
- เมื่อเก็บเป็น float32 eigenvalue นั้นอาจติดลบเล็กน้อยได้ด้วย
- นอกจากนี้ยังมี eigenvalue อีกประมาณ 60 ตัวที่เล็กมาก

**ผลที่ตามมา**
- `ReEig` (ปรับค่าให้อย่างน้อย 1e-4) จำเป็นจริง ๆ ถ้าไม่มี `LogEig` จะเจอ log(0) หรือ log(ค่าติดลบ) แล้วได้ −inf หรือ NaN
- MDRM และ TSLDA ซึ่งใช้ log-Euclidean กับ covariance ตรง ๆ จึงต้องลดมิติด้วย PCA 16 มิติก่อน
- ประโยคพร้อมใช้ใน presentation: *"ทฤษฎีบอกว่า covariance เป็น SPD แต่สำหรับ MedSAM สถาปัตยกรรมเองทำให้มันเป็นแค่ PSD และนี่คือเหตุผลที่ SPDNet ต้องมี ReEig"*

### 1.7 กับดักทั้ง 4 ข้อ และวิธีเลี่ยง ✅

| # | กับดัก | ป้ายในโค้ด | วิธีเลี่ยง |
|---|---|---|---|
| a | `fit_spdnet` โหลดผลเก่ามาใช้เองโดยไม่เทรนใหม่ ถ้ามี `*_metrics.json` อยู่ | `[P1.7-a]` | ลบ `models/SPDNet/<ds>/<gve>/` ก่อนรันใหม่ทุกครั้ง |
| b | cache เช็คแค่จำนวนรูป ไม่สนว่า preprocessing เปลี่ยน | `[P1.7-b]` | ลบ `cache/<ds>/<gve>/` เมื่อเปลี่ยน preprocessing และจด config ของ cache ไว้ |
| c | ไม่มี seed | `[P1.7-c]` | Phase 2.2: เพิ่ม `seed_everything(i)` ต่อรอบ และบันทึกว่าเป็นสิ่งที่เราเพิ่มเอง |
| d | debug mode ใช้ 2 รูปต่อ class | `[P1.7-d]` | ห้ามรายงานผลจาก `results/debug/` |
| (เพิ่ม) | `torch.set_default_device('cuda')` ทำให้ cache ถูกอ่านขึ้น GPU ทั้งหมด | `[P1.7]` | ระวัง VRAM ตอนเทรน |

---

## ความต่างระหว่าง paper กับ code (สรุปรวม)

| เรื่อง | Paper | Code | ใช้แบบไหน |
|---|---|---|---|
| ตัวหารของ covariance | N−1 | N+1 | code (ไม่มีผลจริง) |
| Optimizer ของ BiMap | manifold Adam | Riemannian SGD | code |
| Preprocessing | ไม่ได้ระบุ | per-channel min-max | code |
| ES / checkpoint | ไม่ได้ระบุ | ES ดู val BA, checkpoint ดู val loss | code |
| SPD | "SPD matrices" | จริง ๆ ได้ PSD rank 255 จัดการด้วย ReEig | อธิบายใน presentation |

---
เวลาเริ่ม run_step1.py 10.25 p.m.

---

## Phase 2: Reproduce บน DermaMNIST ✅

### 2.1–2.3 การแก้โค้ดและ dry run

| ไฟล์ | การแก้ | เหตุผล |
|---|---|---|
| `src/main.py` | เพิ่ม `gves=('dino','medsam')` ใน `handle_spdnet_gve` และเปลี่ยนเป็น `for gve in gves` | ให้รันเฉพาะ MedSAM ได้ ค่า default เดิมยังเหมือนต้นฉบับ |
| `src/run_step1.py` | เรียกด้วย `gves=('medsam',)` | |
| `src/experiments/spdnet.py` | `pl.seed_everything(i, workers=True)` ในแต่ละ repeat | ทำให้รันซ้ำได้ (ต้นฉบับไม่มี seed) |

- Dry run แบบ debug ผ่านเมื่อ 15 ต.ค. เวลา 22:18–22:21 (เวลาไทย)
- รันจริงผ่าน PowerShell ด้วย `python run_step1.py *>&1 | Tee-Object ..\..\results\run_step1_medsam.log`
- ข้อความ `NativeCommandError` ใน log เกิดจาก PowerShell มอง stderr ว่าเป็น error ไม่ใช่ error จริง

### 2.4–2.5 เวลาที่ใช้ (RTX 3050 Ti Laptop, 4 GB)

| ขั้นตอน | เวลา (ไทย) | ใช้เวลา |
|---|---|---|
| สกัด feature: train 7,007 รูป | 22:24 → 23:19 | 55 นาที |
| สกัด feature: val 1,003 รูป | → 23:27 | 8 นาที |
| สกัด feature: test 2,005 รูป | → 23:42 | 15 นาที |
| **รวมการสกัด feature** | | **78 นาที (~0.47 s/รูป)**, cache 2.6 GB |
| SPDNet repeat 1 (best epoch 19) | 23:42 → 00:04 | 22 นาที |
| repeat 2 (best epoch 45) | → 00:45 | 41 นาที |
| repeat 3 (best epoch 19) | → 01:11 | 25 นาที |
| repeat 4 (best epoch 23) | → 01:37 | 27 นาที |
| repeat 5 (best epoch 44) | → 02:38 | 60 นาที |
| **รวมการเทรน** | | **2 ชม. 56 นาที** (45–53 วินาทีต่อ epoch, 219 batch) |

- "best epoch" คือ checkpoint ที่ val loss ต่ำสุด ซึ่งเป็นตัวที่ถูกนำไป test
- ทุก repeat หยุดด้วย early stopping นานก่อนจะถึง 250 epoch

### 2.6 ผล (ไฟล์เต็มอยู่ที่ `results/dermamnist_medsam_spdnet/summary.md`)

ผลคำนวณจาก `prediction.npy` ด้วย `analysis/eval_report.py` และตรงกับ `np_metrics.json` ของ official ทุก run (ต่างกันไม่เกิน 2e-16)

| Run | Acc | **Bal. acc** | Macro F1 | **Macro AUROC** |
|---|---|---|---|---|
| 1 | 60.4 | 54.5 | 39.9 | 88.5 |
| 2 | 59.2 | 56.1 | 40.6 | 89.0 |
| 3 | 60.0 | 54.7 | 39.7 | 88.3 |
| 4 | 58.4 | 51.4 | 37.0 | 88.7 |
| 5 | 56.5 | 55.5 | 40.2 | 88.5 |
| **mean ± std** | **58.9 ± 1.6** | **54.4 ± 1.8** | **39.5 ± 1.4** | **88.6 ± 0.3** |

**ผลแยก class (mean ± std ของ 5 runs)**

| Class | Test n | Precision | Recall | F1 |
|---|---|---|---|---|
| akiec | 66 | 25.0 | 51.2 ± 15.5 | 32.3 |
| bcc | 103 | 45.5 | 48.3 ± 11.8 | 45.6 |
| bkl | 220 | 47.7 | 41.7 ± 10.4 | 43.2 |
| df | 23 | 8.3 | 46.1 | 14.0 |
| mel | 223 | 33.1 | 49.5 | 39.4 |
| nv | 1341 | **93.6** | **64.2** | 76.1 |
| vasc | 29 | 15.6 | **80.0** | 25.8 |

ภาพ confusion matrix อยู่ที่ `results/dermamnist_medsam_spdnet/confusion_matrix.png`

**ตีความ (สิ่งที่ข้อมูลแสดง)**
- **Accuracy 58.9% ต่ำกว่าการทายเป็น nv ทุกรูป (67%)** แต่ balanced accuracy 54.4% สูงกว่าแบบนั้นมาก (14.3%)
  - สาเหตุคือ class weight (1/n) ใน loss ซึ่งดันให้โมเดลทาย class เล็กบ่อยขึ้น
  - ผลคือ recall ของ class เล็กสูง (vasc 80%) แต่ precision ต่ำ (vasc 16%, df 8%)
  - ขณะที่ nv ซึ่งเป็น class ใหญ่ เสีย recall ไปบางส่วน (64%)
  - นี่คือ trade-off ที่ตั้งใจไว้ โค้ด optimize เพื่อ balanced accuracy ไม่ใช่ accuracy
- **AUROC สูง (88.6) แต่ BA กลาง ๆ** แปลว่าการจัดลำดับความน่าจะเป็นดี แต่การตัดสินด้วย argmax ยังทำให้ class ปนกันมาก
- คู่ที่ปนกันมาก: bcc→df 17%, akiec→bcc 18%, bkl→akiec 20%, nv→mel 14% และ mel→nv 13%
- recall ของ akiec, bcc และ bkl แกว่งมากระหว่าง run (std 10–15) แปลว่า init ของ W มีผลต่อผลแยก class มาก

### 2.7 เทียบกับ paper ✅

เกณฑ์ที่ตั้งไว้ล่วงหน้าใน ROADMAP (8 ต.ค. ก่อนรัน): ค่าของ paper ต้องอยู่ในช่วง mean ± 2·std หรือ BA ต่างไม่เกินประมาณ 3 จุด

| Metric | ของเรา (5 runs) | Paper | ช่วง mean ± 2·std | ผ่าน? |
|---|---|---|---|---|
| Balanced accuracy | 54.4 ± 1.8 | 56.3 | 50.8 – 58.0 | ✅ (ต่าง 1.9 จุด) |
| Macro AUROC | 88.6 ± 0.3 | 88.5 | 88.0 – 89.2 | ✅ |

**สรุป: reproduce SPDNet + MedSAM บน DermaMNIST ได้**
- BA ของเราต่ำกว่า paper เล็กน้อย แต่อยู่ในช่วงความแปรปรวนระหว่าง run
- สาเหตุที่เป็นไปได้ (ยังไม่ได้ทดสอบ): seed ต่างกัน (paper ไม่ได้ seed), และ GPU หรือ library รุ่นต่างกัน

### สิ่งที่พบระหว่างทาง (ใช้ในการสัมภาษณ์ได้)

1. **`torch_metrics.json` คิด test metrics จาก batch แรก (256 รูป) เท่านั้น** ✅ ยืนยันแล้ว
   - `test/Accuracy` เท่ากับ accuracy ของรูป 256 รูปแรกพอดีทั้ง 5 run (เช่น run 1: 0.56641 = 145/256 ขณะที่ทั้ง test set ได้ 0.60449)
   - ตัวเลขที่รายงาน (CSV และ `np_metrics.json`) คิดจาก sklearn บน test ทั้งชุด **จึงไม่ได้รับผลกระทบ**
   - [P] ค่า `val/...` ใน torch_metrics ก็น่าจะเป็น batch แรกเหมือนกัน (`val/Accuracy` = 0.59765625 = 153/256) ถ้าเป็นจริง **early stopping ซึ่งดู val/Balanced accuracy จะดูจาก val แค่ 256 จาก 1,003 รูป**
   - ยังไม่ได้ยืนยันข้อนี้ และยังไม่รู้สาเหตุในโค้ด (น่าจะอยู่ที่ `_report`/`_reset_metrics` ใน `nn_lit/__init__.py`) ต้องตรวจต่อ
   - **ไม่แก้** เพราะเรากำลัง reproduce ให้เหมือนต้นฉบับ และต้องใช้ protocol เดียวกันกับ Kvasir
2. **CSV ผลติด label ผิด:** แถวขึ้นว่า "BreastMNIST" แต่ตัวเลขเป็นของ DermaMNIST
   - สาเหตุ: `to_dataframe` ใช้ชื่อ dataset ตามลำดับใน list ทั้งหมด (`[ds.display_text for ds in datasets][:len(results)]`) แต่เรารันแค่ dataset เดียว จึงได้ชื่อตัวแรกของ list
   - เป็นแค่ label ผิด ตัวเลขถูก ให้อ้างอิง `results/dermamnist_medsam_spdnet/summary.md` แทน
3. ลำดับ shuffle ของ train เหมือนกันทุก repeat เพราะ `torch.Generator()` ใหม่เริ่มที่ seed คงที่ ความต่างระหว่าง run จึงมาจาก init ของ W เท่านั้น

---

## Phase 3: เตรียม Kvasir-v2 ✅

### 3.1 Audit (`kvasir_prep/audit_kvasir.py` → `splits/kvasir_v2_inventory.csv`)

- 8 class × 1,000 = 8,000 รูป, เปิดไม่ได้ 0 รูป, เป็น RGB ทั้งหมด
- class_to_idx เรียงตามชื่อโฟลเดอร์: dyed-lifted-polyps 0, dyed-resection-margins 1, esophagitis 2, normal-cecum 3, normal-pylorus 4, normal-z-line 5, polyps 6, ulcerative-colitis 7

**ขนาดรูป**

| ขนาด | จำนวน | สัดส่วน (กว้าง:สูง) |
|---|---|---|
| 720×576 | 5,137 | 1.25 |
| 1280×1024 | 2,122 | 1.25 |
| 1920×1072 | 734 | 1.79 |
| อื่น ๆ | 7 | |

**⚠ ความละเอียดผูกกับ class** (เสี่ยงที่โมเดลจะใช้เป็นทางลัด / confound)
- normal-cecum เป็น 720×576 ทั้ง 1,000 รูป
- esophagitis, normal-pylorus, normal-z-line: 1280×1024 อยู่ที่ 65–73% และ 1920×1072 อยู่ที่ 21–25%
- class ที่เหลือเป็น 720×576 อยู่ที่ 97–100%
- แม้หลัง resize เป็น 1024² แล้ว ร่องรอยของความละเอียดเดิมก็อาจยังเหลืออยู่ เช่น ความคม การถูกบีบแนวนอนของรูป 1.79 และกรอบเขียว ต้องเขียนเป็น limitation

**ไฟล์ซ้ำทุก byte (sha1):** 12 คู่
- 1 คู่อยู่ข้าม class: ไฟล์ `2164dff2...jpg` ตัวเดียวกันอยู่ทั้งใน dyed-lifted-polyps และ dyed-resection-margins แปลว่า label ขัดกัน

### 3.2 รูปที่เกือบซ้ำกัน (`kvasir_prep/find_near_duplicates.py`)

**วิธี:** dHash ที่เขียนเอง ตัดกลางภาพ (x 25–85%, y 15–75%) แล้วเปรียบเทียบที่ 16×16 = 256 bit วัดความต่างเป็นระยะ Hamming และเทียบครบทุกคู่

**รอบแรกล้มเหลว:** hash ทั้งภาพที่ 64 bit วัด layout ของหน้าจอ (ขอบดำ ตัวหนังสือ กรอบเขียว) แทนเนื้อเยื่อ
- รูปที่คนละภาพกันได้ระยะ 2
- เจอคู่ข้าม class ถึง 144 คู่
- บทเรียน: ต้อง**ดูภาพยืนยัน**ทุกครั้ง อย่าเชื่อตัวเลขอย่างเดียว

**รอบสอง (ตรวจด้วยตาจาก `splits/near_dup_examples.png`)**

| ระยะ d | สิ่งที่พบ |
|---|---|
| 0–8 | 28 คู่ ซ้ำจริง |
| 9–20 | ไม่มีคู่เลย (ช่องว่างตามธรรมชาติ) |
| 21, 26 | ฉากเดียวกัน (กรณีก้ำกึ่ง) |
| ≥ 22 | ส่วนใหญ่เป็นคนละภาพ |

### 3.3–3.4 การตัดสินใจ (ผู้ทดลองเลือกเอง 16 ต.ค.)

| # | การตัดสินใจ | เหตุผล |
|---|---|---|
| 1 | **70/10/20** แยกต่อ class (stratified) | เหมือน DermaMNIST (7:1:2) จึงเทียบ protocol กันได้; test มี 200 รูปต่อ class |
| 2 | **seed = 42, ใช้ split เดียว** | กำหนดก่อนเห็นผล และห้ามเลือก seed ตามผล เพราะเท่ากับเลือก test set ที่ง่าย (selection bias); ความแปรปรวนวัดจาก 5 training repeats |
| 3 | **จับกลุ่มรูปที่ d ≤ 20 ให้อยู่ split เดียวกัน และตัดกลุ่มที่ label ขัดกัน** | กัน leakage โดยไม่ทิ้งข้อมูล; 20 คือจุดที่มีช่องว่างตามธรรมชาติในข้อมูล; คู่ที่ label ขัดกันไม่รู้ว่า label ไหนถูก จึงตัดทิ้งทั้งสองไฟล์ |
| 4 | **resize ตรงเป็น 1024²** ด้วย `transform_demo` เดิม | ใช้โค้ดเดิมของ official โดยไม่ต้องแก้ (เป็นการเปลี่ยนให้น้อยที่สุด); limitation คือรูป 1.79 ถูกบีบ และสัมพันธ์กับ class |

### ผล split (`kvasir_prep/make_split.py`)

**วิธี:**
1. ใช้ union-find จับรูปซ้ำเข้ากลุ่ม
2. สุ่มลำดับ "กลุ่ม" ทีละ class ด้วย `np.random.default_rng(42)`
3. เติมกลุ่มลง test จนครบ 20% แล้ว val จนครบ 10% ที่เหลือเป็น train

**จำนวน:**
- รวม 7,998 รูป (ตัดคู่ที่ label ขัดกันออก 2 รูป) และมีกลุ่มรูปซ้ำ 27 กลุ่ม
- **train 5,598 / val 800 / test 1,600** หรือ 700/100/200 ต่อ class (dyed-lifted-polyps และ dyed-resection-margins มี train 699)
- assert ผ่าน: ไม่มีกลุ่มรูปซ้ำข้าม split และไม่มีรูปอยู่สอง split

**ยืนยันว่ารันซ้ำได้ผลเดิม:** รัน 2 ครั้งได้ sha256 ของ `splits/kvasir_v2_split.csv` เท่ากัน

```
4b477c457e726bc7f201fe7fab04e6ec55c324e6941a1acc6791f9b3e679ad1f
```

พารามิเตอร์ทั้งหมดอยู่ใน `splits/kvasir_v2_split_info.json`

**กฎ:** ทุกการทดลองบน Kvasir (SPDNet, GAP และ ablation ทุกตัว) ต้องอ่าน split จากไฟล์นี้เท่านั้น

### 3.5 Dataset class (`2026-isbi-mayr/src/kvasir_dataset.py`) ✅

- `KvasirDataset(root, split_csv, split)` คืน `ds[i]` ในรูปแบบเดียวกับ MedMNIST คือ `(np.float32 (3,H,W) ในช่วง [0,1], np.int64 shape (1,))` จึงส่งเข้า `default_transform`, `data_mapper` และ `CachedDataset` ของ official ได้โดยไม่ต้องแก้
- **ไม่** resize หรือ normalize ในไฟล์นี้ เพราะ `transform_demo` ทำให้แล้ว เหมือนกับ DermaMNIST
- ลำดับรูปเรียงตาม CSV ซึ่งเรียงตาม relpath ไว้แล้ว จำเป็นต่อ cache เพราะ cache อ้างอิงข้อมูลตาม index
- `kvasir_info()` คืน `dataset_name='kvasir_v2'` และ `labels` 8 class ใช้แทน MedMNIST2D enum ตอนเรียก `fit_spdnet`
- `train_class_counts()` คืนจำนวนรูปต่อ class ใน train ได้ 699/699/700×6 จึงทำให้ class weight แทบเท่ากันทุก class
- **ผลทดสอบ**
  - split มี 5,598/800/1,600 รูป
  - `test[0]` ได้ (3,576,720) float32 อยู่ในช่วง [0,1] และ label shape (1,)
  - รูป 1280×1024 ผ่าน `default_transform` → MedSAM ได้ feature (256,64,64), covariance (256,256) ที่สมมาตร และ GAP (256)
  - GAP เท่ากับค่าเฉลี่ยที่ `bcov` ลบทิ้งพอดี ✅

---

## Phase 4: Kvasir ด้วย covariance + SPDNet ✅

### 4.2–4.3 สกัด feature ✅ (`src/run_kvasir_extract.py`)

- รัน MedSAM ครั้งเดียวต่อรูป แล้วเก็บ `covs` (256,256), `gaps` (256,) และ `labels` ลง `cache/kvasir_v2/medsam/<split>/`
- ใช้โค้ด official ทั้งหมด (`default_transform` → `transform_demo` → MedSAM encoder → `cov_descr`) ต่างจากต้นฉบับเฉพาะตรงที่ mapper คืนค่า 3 คอลัมน์
- ปัญหาที่เจอ: PowerShell ใช้ code page cp874 ซึ่งไม่มีอักขระ `→` และ `✓` ทำให้ `print` crash แก้โดยเปลี่ยนเป็นสัญลักษณ์ ASCII และใส่ `errors='replace'`

**เวลาและขนาด (RTX 3050 Ti Laptop)**

| split | รูป | เวลา | s/รูป |
|---|---|---|---|
| train | 5,598 | 45.6 นาที | 0.49 |
| val | 800 | 7.5 นาที | 0.56 |
| test | 1,600 | 13.5 นาที | 0.51 |
| **รวม** | | **66.6 นาที** | cache 2.1 GB |

**ผลตรวจ:** label ตรงกับ split CSV ✓, covariance สมมาตร ✓, eigenvalue ต่ำสุด ≈ 0 (rank 255) ✓

### 4.4 สคริปต์เทรน (`src/run_kvasir_train.py`)

- ทั้งสองวิธีเรียก `fit_spdnet` ตัวเดียวกัน → protocol เหมือนกันทุกอย่าง ต่างกันแค่ head
  - SPDNet: BiMap 256→128→64→32 → LogEig → Linear(1024, 8) มี parameter ประมาณ 51,208
  - GAP: [z-score] → Linear(256, 8) มี parameter 2,056
- สำหรับ GAP สคริปต์จะสลับ `BasicSPDNet` เป็น `GapLinear` ชั่วคราว ไม่ต้องเขียน training loop ใหม่
- อ่าน cache แบบอ่านอย่างเดียว เพราะ `CachedDataset` อาจเขียนทับไฟล์ถ้ามันคิดว่า cache เสีย

## Phase 5: GAP baseline

### 5.3 การตัดสินใจเรื่อง z-score ✅ (ผู้ทดลองเลือก 16 ต.ค. ก่อนเห็นผล)

- **baseline หลัก = GAP + z-score + Linear**
  - mean/std คำนวณจาก train เท่านั้น
  - clamp std ไว้ที่ 1e-6 กันหารด้วยศูนย์
- **ablation = GAP แบบไม่ z-score + Linear**
- **หลักฐานที่ใช้ตัดสินใจ:**
  - ค่า GAP เล็ก (std รวมประมาณ 0.1)
  - std ต่อมิติต่างกันประมาณ 1,900 เท่า บางมิติแทบเป็นค่าคงที่
  - nearest-centroid บน train: แบบดิบ 61.8% เทียบกับ z-score 66.4% (เป็นแค่สัญญาณเบื้องต้น ไม่ใช่ผลจริง)
- **เหตุผล:** baseline ที่แข็งแรงทำให้การเปรียบเทียบยุติธรรมต่อสมมติฐาน "covariance ดีกว่า GAP" และกันข้อโต้แย้งว่า GAP แพ้เพราะ optimization ไม่ดี
- **กฎ:** ต้องรายงานทั้งสองแบบ และห้ามสลับว่าตัวไหนเป็นตัวหลักหลังเห็นผล

---

## ผล Kvasir-v2 (เทรนคืน 16 ต.ค.) ✅

ทั้งสามวิธีใช้ split เดียวกัน (test 1,600 รูป) feature จากการรัน MedSAM รอบเดียวกัน และ training loop เดียวกัน เทรนวิธีละ 5 ครั้ง

| วิธี | Balanced acc (= acc, เพราะทุก class มีจำนวนเท่ากัน) | Macro F1 | Macro AUROC | เวลาเทรน (5 ครั้ง) | จำนวน parameter |
|---|---|---|---|---|---|
| **Covariance + SPDNet** | **85.3 ± 0.8** | 85.2 ± 0.8 | 98.8 ± 0.1 | 131 นาที | 51,208 |
| **GAP + z-score + Linear** (baseline หลัก) | **84.8 ± 0.4** | 84.8 ± 0.4 | 98.7 ± 0.0 | 11 นาที | 2,056 |
| GAP แบบไม่ z-score + Linear (ablation) | 75.0 ± 3.1 | 74.8 ± 3.3 | 97.0 ± 0.7 | 26 นาที | 2,056 |

### ความต่างมีนัยสำคัญไหม? (bootstrap บน test set เดียวกัน, 10,000 รอบ)

| คู่ที่เปรียบเทียบ | ต่างกัน | ช่วงความเชื่อมั่น 95% | สรุป |
|---|---|---|---|
| SPDNet − GAP z-score | +0.5 จุด | [−0.9, +2.0] | **ไม่ต่างอย่างมีนัยสำคัญ** (ช่วงคร่อม 0) |
| SPDNet − GAP ไม่ z-score | +10.3 จุด | [+8.5, +12.0] | ต่างจริง |
| GAP z-score − GAP ไม่ z-score | +9.8 จุด | [+8.0, +11.5] | ต่างจริง: การปรับสเกลสำคัญมาก |

**การวิเคราะห์เพิ่มเติม (exploratory ไม่ได้วางแผนไว้ก่อน จึงห้ามใช้เป็นผลหลัก):**
- ถ้ารวมผลของ 5 โมเดลด้วยการเฉลี่ยความน่าจะเป็น (ensemble) SPDNet ได้ 87.1% ส่วน GAP z-score ได้ 85.0% (McNemar p = 0.015)
- เหตุผล: GAP + Linear เป็นปัญหาที่มีคำตอบดีที่สุดคำตอบเดียว ทั้ง 5 ครั้งจึงได้โมเดลแทบเหมือนกัน (std 0.4) การรวมจึงไม่ช่วยอะไร ส่วน SPDNet แต่ละครั้งเริ่มจากค่าสุ่มต่างกันและได้คำตอบหลากหลายกว่า (std 0.8) การรวมจึงช่วยได้
- protocol หลักของ paper คือ "เฉลี่ยผลของ 5 ครั้ง" ไม่ใช่ ensemble

### แยกตาม class (recall เฉลี่ย 5 ครั้ง: SPDNet เทียบกับ GAP z-score)

- คู่ที่สับสนกันมากที่สุดเหมือนกันทั้งสองวิธี:
  - esophagitis ↔ z-line (ทั้งคู่เป็นบริเวณรอยต่อหลอดอาหาร)
  - dyed-lifted ↔ dyed-resection (ทั้งคู่ย้อมสีฟ้า)
- SPDNet ดีกว่าที่ z-line (82.5 เทียบกับ 76.3) แต่**แย่กว่า**ที่ esophagitis (74.8 เทียบกับ 79.5) แปลว่าเป็นการ "เลื่อนเส้นแบ่ง" ระหว่างคู่เดิม มากกว่าจะ "แยกได้ดีขึ้นจริง"
- class ที่ง่าย (pylorus, cecum, UC, polyps) ได้ผลเท่ากันทั้งสองวิธี

### ต้นทุน

| | GAP + Linear | SPDNet |
|---|---|---|
| ให้ MedSAM ดูรูป (ทั้งสองวิธีเท่ากัน และเป็นต้นทุนหลัก) | ~0.5 s/รูป, 67 นาที | ~0.5 s/รูป, 67 นาที |
| ขนาดข้อมูลต่อรูป | 1 KB | 256 KB (×256) |
| จำนวน parameter | 2,056 | 51,208 (×25) |
| เวลาเทรน | 11 นาที | 131 นาที (×12) |

### คำตอบเบื้องต้นสำหรับคำถามของอาจารย์ข้อ 3 (Kvasir)

- บน Kvasir วิธี covariance + SPDNet **ไม่ได้เปรียบอย่างมีนัยสำคัญ** เมื่อเทียบกับ GAP ที่ปรับสเกลแล้ว (85.3 เทียบกับ 84.8) แต่ใช้เวลาเทรนมากกว่า 12 เท่า
- **ยังขาดชิ้นส่วนหนึ่ง:** เราไม่มี GAP baseline บน DermaMNIST จึงยังบอกไม่ได้ว่า "ความได้เปรียบหายไปเมื่อย้ายมา Kvasir" หรือ "ไม่เคยได้เปรียบตั้งแต่แรก" ถ้าต้องการตอบข้อ 3 ให้ครบ ควรรัน GAP บน Derma ด้วย

---

## ผล GAP baseline บน DermaMNIST (เพื่อตอบคำถามอาจารย์ข้อ 3) ✅

**การสกัด feature:** ใช้ `run_derma_extract.py` ใช้เวลา 72 นาที
- label ตรงกับ DermaMNIST ทุกชุด
- **covariance ตรงกับ cache official ที่ใช้ใน Phase 2 พอดี (ต่างกัน 0.0)** จึงพิสูจน์ได้ว่าเป็น feature ชุดเดียวกับที่ใช้เทรน SPDNet

### ตารางรวม 2 dataset × 2 วิธี (แต่ละวิธีเทรน 5 ครั้ง)

| Dataset | วิธี | Balanced acc | Macro AUROC | เวลาเทรน |
|---|---|---|---|---|
| DermaMNIST (64×64 ขยายเป็น 1024) | Covariance + SPDNet | 54.4 ± 1.8 | **88.6 ± 0.3** | ~176 นาที |
| | GAP + z-score + Linear | **56.0 ± 0.8** | 87.3 ± 0.3 | 5 นาที |
| | GAP ไม่ z-score (ablation) | 45.6 ± 0.4 | 81.8 ± 0.0 | 13 นาที |
| Kvasir-v2 (รูปความละเอียดจริง) | Covariance + SPDNet | 85.3 ± 0.8 | 98.8 ± 0.1 | 131 นาที |
| | GAP + z-score + Linear | 84.8 ± 0.4 | 98.7 ± 0.0 | 11 นาที |
| | GAP ไม่ z-score (ablation) | 75.0 ± 3.1 | 97.0 ± 0.7 | 26 นาที |

### SPDNet − GAP z-score (bootstrap บน test set เดียวกัน, ช่วงความเชื่อมั่น 95%)

| Dataset | Balanced acc | Macro AUROC |
|---|---|---|
| DermaMNIST | −1.5 [−4.8, +1.8] → ไม่ต่าง | **+1.3 [+0.3, +2.5] → SPDNet ดีกว่าจริง** |
| Kvasir-v2 | +0.5 [−0.9, +1.9] → ไม่ต่าง | +0.05 [−0.1, +0.2] → ไม่ต่าง |

### สรุป (สิ่งที่ข้อมูลแสดง)

1. **"ตัดสินถูกกี่ข้อ" (balanced accuracy):** ทั้งสอง dataset ไม่มีวิธีไหนดีกว่าอย่างมีนัยสำคัญ บน Derma วิธีค่าเฉลี่ยได้ 56.0 ซึ่งเท่ากับตัวเลขที่ paper รายงานของ SPDNet (56.3)
2. **"จัดลำดับความน่าจะเป็นได้ดีแค่ไหน" (AUROC):** SPDNet ดีกว่าเล็กน้อยแต่มีนัยสำคัญ**เฉพาะบน Derma** และความได้เปรียบนี้**หายไปบน Kvasir**
3. การปรับสเกลค่าเฉลี่ย (z-score) สำคัญมากทั้งสอง dataset (+10 จุด) ถ้าไม่ปรับ วิธีของ paper จะดูเหมือนชนะห่าง ซึ่งไม่ยุติธรรม
4. ต้นทุน: SPDNet ใช้เวลาเทรนมากกว่า 12–35 เท่า ส่วนการให้ MedSAM ดูรูปเท่ากันทั้งสองวิธี

### สมมติฐานที่อธิบายได้ (ยังไม่ได้พิสูจน์)

- รูป Derma เล็ก (64×64) แล้วถูกขยาย 16 เท่า ทำให้แต่ละช่องในรูปเกือบเหมือนกัน "ความสัมพันธ์ระหว่าง feature" จึงเป็นข้อมูลเสริมเล็ก ๆ ที่ช่วยจัดลำดับได้
- รูป Kvasir คมชัด และ MedSAM แยกประเภทได้ดีมากอยู่แล้ว ค่าเฉลี่ยจึงพอ (AUROC ประมาณ 99 ทั้งคู่ เหลือที่ให้ปรับปรุงน้อยมาก)
- ข้อผิดพลาดที่เหลือบน Kvasir เป็นคู่ที่หน้าตาคล้ายกันจริง ๆ (esophagitis/z-line และ dyed-lifted/dyed-resection) ซึ่งทั้งสองวิธีผิดเหมือนกัน
