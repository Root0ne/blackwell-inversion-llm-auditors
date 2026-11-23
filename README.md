# Hesaplamalı Blackwell Tersinmesi: Deney Kodu ve Ham Kayıtlar

Bu depo, **"Araç Destekli Büyük Dil Modeli Denetçilerinde Gözlemciye Göre Sanal Kara-Kutu Bariyerleri ve Hesaplamalı Blackwell Tersinmesi"** başlıklı makalenin deney kodunu, 7.631 canlı denemenin ham kayıtlarını ve analiz betiklerini içerir. Makale hakem değerlendirmesindedir.

*English summary.* Code, raw logs and analysis scripts for a study of tool-using LLM agents acting as security auditors. The study measures when giving an agent white-box access (source code or neural-network weights) on top of black-box query access makes it *less* accurate than black-box access alone under the same interaction budget. Code comments and docstrings are in English; this README and the protocol note are in Turkish.

## Çalışma neyi ölçüyor?

Bir denetçi ajan iki erişim kipinde, aynı tohumlar ve aynı bütçe altında sınanır:

- **Kara-kutu (BB):** ajan sistemi yalnızca girdi gönderip çıktı okuyarak inceler.
- **Beyaz-kutu (WB):** ajan aynı sorgu arayüzüne ek olarak sistemin kaynak kodunu veya sinir ağı ağırlıklarını da görür.

İki kip arasındaki başarı farkı `Delta_WB = WB - BB` olarak raporlanır. Depodaki iki derleyici bu farkın sıfıra yakın kaldığı ve negatife döndüğü koşulları, sistemin dış davranışını değiştirmeden üretir:

- **Deney 1 (`exp1`, `C_kXOR`):** gürültülü k-XOR denklemlerine gömülü bir arka kapının varlığını ayırt etme görevi.
- **Deney 2 (`exp2`, `C_inv`):** gizli bir Boole kuralını geri çıkarma görevi; beyaz-kutu temsili, hiçbir girdide çalışmayan yanıltıcı bir tuzak dalı içerir.

## Dosyalar

Tüm kod `pilot/` paketindedir.

| Dosya | İçerik |
| --- | --- |
| `generators.py` | İki derleyici: `C_kXOR` örnek üreteci (Python kodu, MBA karartmalı kod, 2 ve 3 katmanlı ReLU ağırlıkları) ve `C_inv` üreteci (beş beyaz-kutu varyantı). Alarm eşiği ve yanlış alarm üst sınırı burada hesaplanır. |
| `validators.py` | Üreteçlerden bağımsız doğrulama: dışa aktarılan kod ve ağırlıkların tüm varyantlarda aynı çıktıyı verdiğini ve bayt-özdeş kontrol çiftinin eşit uzunlukta olduğunu denetler. |
| `baselines.py` | Sekiz saldırılık klasik referans bataryası (ağırlık istatistikleri, SVD, Canon+Kikuchi spektral çözücü, benzetimli tavlama, WalkSAT) ve kara-kutu avantaj üst sınırı. |
| `sandbox.py` | Ajanın yazdığı Python kodunu çalıştıran yalıtılmış alt süreç (`IsolatedPythonSandbox`) ve sorgu kotasını uygulayan arayüz (`MeteredOracle`). |
| `runner.py` | Deney yürütücüsü. Anthropic ve Google Gemini API istemcileri, araç çağrısı döngüsü, maliyet tavanı ve kaldığı yerden devam eden JSONL kaydı. |
| `analyze.py` | İstatistik işlevleri: kesin Fisher testi, Newcombe–Wilson güven aralığı, eşleştirilmiş permütasyon testi, bootstrap, Holm–Bonferroni düzeltmesi. |
| `analyze_c3_unified.py` | Üç kayıt dosyasını birlikte analiz eder ve makaledeki tabloların dayandığı `report_c3_unified.json` raporunu üretir. |
| `make_figures_tr.py` | Makaledeki şekilleri `report_c3_unified.json` dosyasından üretir. |
| `make_figures.py` | Şekillerin önceki, dokuz modelli İngilizce sürümünü üretir. |
| `test_pilot.py` | 16 birim ve bütünleşik test. |
| `on_kayit.md` | Deneylerden önce yazılan yerel çalışma protokolü: hipotezler, tohum dizileri ve karar kuralları. |
| `results_*.jsonl` | Ham deneme kayıtları (aşağıya bakınız). |
| `report_*.json`, `report_*.md`, `report_*.txt` | Analiz betiklerinin ürettiği raporlar. |

## Kurulum

Python 3.14.4, numpy 2.3.5, scipy 1.16.3 ve matplotlib 3.10.7 ile çalıştırılmıştır. Yürütücü yalnızca standart kütüphaneyi kullanır; API için ayrı bir paket gerekmez.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Yalıtılmış alt süreç Linux kaynak sınırlarını (`resource` modülü) kullandığı için kod Linux üzerinde çalıştırılmalıdır.

## API anahtarı gerektirmeyen doğrulama

Aşağıdaki üç adım hiçbir API çağrısı yapmaz. Komutlar depo kök dizininden çalıştırılır.

```bash
# 1) Birim ve bütünleşik testler (16 test, yaklaşık 1,5 dakika)
python3 -m unittest pilot.test_pilot

# 2) Makaledeki istatistikleri ham kayıtlardan yeniden hesaplama
python3 -m pilot.analyze_c3_unified --out-json /tmp/report.json > /tmp/report.txt
cmp /tmp/report.json pilot/report_c3_unified.json   # çıktı vermemesi, raporun birebir aynı olduğunu gösterir

# 3) Deney hattını sahte istemciyle uçtan uca çalıştırma (maliyet 0 USD)
python3 -m pilot.runner --dry-run --experiment exp2 --models claude-haiku-5-5 --exp2-seeds 1 --out /tmp/dry.jsonl
```

Şekilleri yeniden üretmek için:

```bash
python3 -m pilot.make_figures_tr   # çıktı: paper_tr/figures/
```

## Deneyleri canlı çalıştırma

Canlı koşular ücretlidir. Anahtarlar ortam değişkeninden okunur ve depoda saklanmaz:

```bash
export ANTHROPIC_API_KEY=...   # Claude modelleri
export GEMINI_API_KEY=...      # Gemini ve Gemma modelleri
python3 -m pilot.runner --experiment exp2 --models claude-haiku-5-5 \
    --effort medium --exp2-seeds 25 --max-cost-usd 5 --out results_yeni.jsonl
```

`--max-cost-usd` harcama tavanıdır; tavana ulaşıldığında yürütücü durur. Aynı `--out` dosyasıyla yeniden başlatılan koşu, tamamlanmış denemeleri atlar. Tüm seçenekler için: `python3 -m pilot.runner --help`.

## Ham kayıtlar

Makaledeki 7.631 deneme üç dosyadadır. Her satır bir denemedir.

| Dosya | Deneme | İçerik | SHA-256 |
| --- | ---: | --- | --- |
| `results_c1_haiku55.jsonl` | 680 | Parametrik zorluk kalibrasyonu (Haiku 5.5) | `045dc5c6c30d7e88e3de6efcaa57be466b7bf8c00a9b06fe9062076509fb1b00` |
| `results_c2_pilot.jsonl` | 5.026 | Dokuz Anthropic modeli: 2.064 Deney 1 ve 2.962 Deney 2 denemesi | `ba07b3d6c6a31bf478b04130fc062c06b0d74217511e8e923f69bb4a9146c6fc` |
| `results_c3_expansion.jsonl` | 1.925 | Google Gemini, Gemma 4 ve nedensel kontroller: 508 Deney 1 ve 1.417 Deney 2 denemesi | `5800f0c81315ff5b6e88303d834997899a815882eab26ec22009ec8134e498c6` |

`results_c3_expansion.jsonl`, aynı klasördeki dokuz kaynak dosyanın (`results_c3_anth_key*.jsonl`, `results_c3_gemini*.jsonl`, `results_c3_gemma*.jsonl`) `analyze_c3_unified.py --write-merged` ile birleştirilmiş hâlidir; kaynak dosyalar izlenebilirlik için depoda bırakılmıştır. `results_c3_gemini3_flash_free.jsonl` dosyasındaki üç kayıt, makalede yer almayan bir modelle yapılmış arayüz denemesidir ve 7.631 toplamına dâhil değildir.

Bir kayıttaki başlıca alanlar:

| Alan | Anlamı |
| --- | --- |
| `run_key` | Denemenin benzersiz anahtarı (deney, model, efor, bütçe profili, kip, varyant, tohum) |
| `model`, `effort`, `thinking_mode` | Model kimliği ve akıl yürütme ayarı |
| `experiment`, `family`, `regime`, `substrate`, `variant`, `mode` | Deney koşulu |
| `n`, `m`, `k`, `eta`, `seed` | Örnek parametreleri ve tohum |
| `budget_track`, `turns`, `queries_used`, `q_max`, `python_calls` | Bütçe profili ve harcanan etkileşim |
| `parse_ok`, `truncated_turns`, `stop_reasons` | Yanıtın geçerli JSON olup olmadığı ve token sınırında kesilme |
| `decision_label`, `correct` | Deney 1 kararı ve doğruluğu |
| `pred_support`, `pred_signs`, `heldout_acc`, `matches_decoy` | Deney 2 tahmini, 64 test noktasındaki doğruluk ve tuzak kuralla eşleşme |
| `input_tokens`, `output_tokens`, `thinking_tokens`, `cost_usd`, `wall_s` | Token, maliyet ve süre ölçümleri |

## Değerlendirilen modeller

- **Anthropic (9):** `claude-sonnet-4-5-20250929`, `claude-haiku-4-5-20251001`, `claude-opus-4-6`, `claude-sonnet-4-6`, `claude-opus-4-8`, `claude-haiku-5-5`, `claude-sonnet-5-5`, `claude-opus-5-5`, `claude-fable-5-2`
- **Google Gemini (4):** `gemini-3.1-flash-lite`, `gemini-3.5-flash-lite`, `gemini-3.8-flash`, `gemini-4-carbon`
- **Açık ağırlıklı Gemma 4 (2):** `gemma-4-26b-a4b-it`, `gemma-4-31b-it`

## Bilinmesi gerekenler

- `on_kayit.md` yerel bir protokol belgesidir; harici bir platformda zaman damgalı ön-kayıt yapılmamıştır. Belgedeki bazı bağlantılar yazarın yerel dosya yollarını gösterir ve depoda açılmaz.
- Ana Deney 1 ailesi olan `4xor_strict` parametreleri, kalibrasyon aşamasındaki bulgulardan sonra belirlenmiştir.
- API sağlayıcıları model sürümlerini değiştirebildiği için canlı koşular kayıtlardaki sayıları birebir vermeyebilir. Kayıtlardan yapılan analiz ise deterministiktir.

## Atıf

Makale yayımlandığında atıf bilgisi buraya eklenecektir.
