# Pulat

Kendine karşı oynayarak öğrenen tavla yapay zekâsı. Ağa hiçbir strateji bilgisi verilmez: sadece oyunun kurallarını bilir, ham tahtaya bakar ve milyonlarca oyunu kendine karşı oynayarak (TD(λ), TD-Gammon yöntemi) öğrenir.

*A backgammon AI that learns purely from self-play (TD(λ), TD-Gammon style). No hand-crafted strategy; Turkish scoring: win = 1 point, gammon (mars) = 2 points, no doubling cube.*

## Özellikler

- **Kural motoru** (`engine.py`): tüm yasal hamleler; en çok zarı oynama ve tek zarda büyüğü oynama kuralları dahil.
- **Hızlı Rust motoru** (`fastengine/`): aynı kurallar, tüm CPU çekirdeklerinde toplu hamle üretimi (saniyede ~2 milyon pozisyon).
- **Sinir ağı** (`model.py`): 196 girdi, 2×256 gizli katman; çıktı: kazanma, mars yapma, mars olma olasılıkları.
- **Eğitim**: `train.py` (saf Python) ve `train_fast.py` (Rust + GPU, binlerce oyun aynı anda).
- **İleriye bakma** (`search.py`): rakibin 21 zarının hepsine bakan 2-ply arama.
- **Oyna**: terminalde (`play.py`) ya da tarayıcıda tıklayarak (`web.py`).
- **Ölçüm**: rastgele oyuncuya, eski sürümlere ve GNU Backgammon'a karşı karşılaştırma araçları.

Hazır eğitilmiş model: `checkpoints/pulat_best.pt`. Bu model rastgele oyuncuyu %99,9 yener ve 15 açılış zarının 13'ünde uzmanların kabul ettiği hamleyi seçer (`opening_test.py`).

## Kurulum

```bash
pip install -r requirements.txt
cd fastengine && cargo build --release && cd ..
```

Rust motoru `train_fast.py`, `web.py`, `search.py` ve ölçüm araçları için gerekir. `engine.py`, `play.py` ve `train.py` Rust olmadan da çalışır. CUDA varsa GPU kullanılır, yoksa CPU.

## Kullanım

```bash
python play.py                         # terminalde Pulat'a karşı oyna
python web.py                          # tarayıcıda oyna: http://localhost:8765
python train_fast.py --hours 3         # sıfırdan hızlı eğitim (checkpoints_fast/)
python train.py --hours 2              # saf Python eğitim (checkpoints/)
python opening_test.py checkpoints/pulat_best.pt
python compare_ply.py checkpoints/pulat_best.pt 1000
python test_fast.py                    # Rust ve Python motorları aynı hamleleri mi üretiyor
```

### GNU Backgammon ile karşılaştırma

`vs_gnubg.py`, Pulat'ı GNU Backgammon'a karşı oynatır ve hamle kalitesini (PR) ölçer. GNU Backgammon kurulu olmalı. `gnubg-cli` yolu `GNUBG` ortam değişkeniyle verilebilir:

```bash
python vs_gnubg.py --model checkpoints/pulat_best.pt --games 1000 --depth 2 --gnubg-plies 2
```

## Tahta biçimi

Durum 28 sayıdır ve her zaman hamle sırası gelen oyuncunun gözündendir:

| İndeks | Anlamı |
|---|---|
| 0..23 | haneler: `+` benim pullarım, `−` rakibin pulları. Ben 23'ten 0'a ilerlerim, evim 0..5 |
| 24 | benim kırık pullarım (bar) |
| 25 | rakibin kırık pulları |
| 26 | benim topladığım pullar |
| 27 | rakibin topladığı pullar |

Ağ, hamleyi yeni yapmış oyuncunun gözünden bakılan hamle sonrası tahtayı değerlendirir. Beklenen puan: `2·kazanma − 1 + mars_yapma − mars_olma`.

## Lisans

MIT
