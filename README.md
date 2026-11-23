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
| `runner.py` | Deney yürütücüsü. Anthropic ve Google Gemini API istemcileri, araç çağrısı döngüsü, maliyet tavanı ve kaldığı yerden devam eden JSONL kaydı. Deney 2 için `--protocol` seçeneğiyle üç önlem protokolü (`verify_first`, `harness_check`, `harness_check_tools`) ve `--log-tool-inputs` ile araç girdilerinin kaydı da buradadır. |
| `analyze.py` | İstatistik işlevleri: kesin Fisher testi, Newcombe–Wilson güven aralığı, eşleştirilmiş permütasyon testi, bootstrap, Holm–Bonferroni düzeltmesi. |
| `analyze_c3_unified.py` | Üç kayıt dosyasını birlikte analiz eder ve makaledeki tabloların dayandığı `report_c3_unified.json` raporunu üretir. |
| `analyze_mechanism.py` | Deney 2 için üç ek analiz: zor koruyuculu kod ile çürütülebilir kontrolün eşdeğerlik analizi, araç çağrısı kayıtlarından ajan davranışı ve yanıtı sorgu arayüzüyle denetleyen karar kuralının kayıtlar üzerinde yeniden oynatımı. `report_mechanism.json` ve `report_mechanism.txt` dosyalarını üretir. |
| `analyze_followup.py` | Veri kümesi kilitlendikten sonra yürütülen ek koşuların analizi: çürütülebilir kontrolün dört modele genişletilmesi, üç protokolün karşılaştırması, araç kullanımına izin veren yanıt denetimi, ajanın kendi sorgularının yanıtıyla çelişip çelişmediği ve tuzak yanıtların ajan tarafından tahmin olarak sunulup sunulmadığı. `report_followup.json` ve `report_followup.txt` dosyalarını üretir. |
| `make_figures_tr.py` | Makaledeki şekilleri `report_c3_unified.json` ve `report_followup.json` dosyalarından üretir. |
| `make_figures.py` | Şekillerin önceki, dokuz modelli İngilizce sürümünü üretir. |
| `test_pilot.py` | 18 birim ve bütünleşik test. |
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
# 1) Birim ve bütünleşik testler (18 test, yaklaşık 1,5 dakika)
python3 -m unittest pilot.test_pilot

# 2) Makaledeki istatistikleri ham kayıtlardan yeniden hesaplama
python3 -m pilot.analyze_c3_unified --out-json /tmp/report.json > /tmp/report.txt
cmp /tmp/report.json pilot/report_c3_unified.json   # çıktı vermemesi, raporun birebir aynı olduğunu gösterir
python3 -m pilot.analyze_mechanism --out-json /tmp/mechanism.json > /tmp/mechanism.txt
cmp /tmp/mechanism.json pilot/report_mechanism.json
python3 -m pilot.analyze_followup --out-json /tmp/followup.json > /tmp/followup.txt
cmp /tmp/followup.json pilot/report_followup.json

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

Deney 2'de `--protocol verify_first` ajana adayını sorgu arayüzünde sınamasını söyler; `--protocol harness_check` ise ajanın yanıtını on yeni sorguyla denetler ve uyuşmazlıkta bir düzeltme turu verir. `--log-tool-inputs`, ajanın yazdığı Python kodunu ve sorguladığı girdileri kayda ekler. Bu protokollerle yapılan koşular makaledeki 7.631 denemenin içinde değildir; ek koşular olarak ayrı dosyalarda tutulur (aşağıya bakınız).

`--max-cost-usd` harcama tavanıdır; tavana ulaşıldığında yürütücü durur. Aynı `--out` dosyasıyla yeniden başlatılan koşu, tamamlanmış denemeleri atlar. Tüm seçenekler için: `python3 -m pilot.runner --help`.

## Ham kayıtlar

Makaledeki 7.631 deneme üç dosyadadır. Her satır bir denemedir.

| Dosya | Deneme | İçerik | SHA-256 |
| --- | ---: | --- | --- |
| `results_c1_haiku55.jsonl` | 680 | Parametrik zorluk kalibrasyonu (Haiku 5.5) | `045dc5c6c30d7e88e3de6efcaa57be466b7bf8c00a9b06fe9062076509fb1b00` |
| `results_c2_pilot.jsonl` | 5.026 | Dokuz Anthropic modeli: 2.064 Deney 1 ve 2.962 Deney 2 denemesi | `ba07b3d6c6a31bf478b04130fc062c06b0d74217511e8e923f69bb4a9146c6fc` |
| `results_c3_expansion.jsonl` | 1.925 | Google Gemini, Gemma 4 ve nedensel kontroller: 508 Deney 1 ve 1.417 Deney 2 denemesi | `5800f0c81315ff5b6e88303d834997899a815882eab26ec22009ec8134e498c6` |

`results_c3_expansion.jsonl`, aynı klasördeki dokuz kaynak dosyanın (`results_c3_anth_key*.jsonl`, `results_c3_gemini*.jsonl`, `results_c3_gemma*.jsonl`) `analyze_c3_unified.py --write-merged` ile birleştirilmiş hâlidir; kaynak dosyalar izlenebilirlik için depoda bırakılmıştır. `results_c3_gemini3_flash_free.jsonl` dosyasındaki üç kayıt, makalede yer almayan bir modelle yapılmış arayüz denemesidir ve 7.631 toplamına dâhil değildir.

### Ek koşular (7.631 denemenin dışında)

Ana veri kümesi kilitlendikten sonra, 23 Kasım 2026'da üç ek deney yürütülmüştür. Üçü de Deney 2'nin ana koşulunu (Çoğunluk-3, 5 tur, medium efor) ve 5000–5024 tohumlarını kullanır. Bu kayıtlarda ajanın yazdığı Python kodu (`tool_log`) ve sorguladığı girdiler (`oracle_history`) da bulunur.

| Dosya | Deneme | İçerik | SHA-256 |
| --- | ---: | --- | --- |
| `results_c4_refutable.jsonl` | 100 | Çürütülebilir kontrolün, ana kümede bu kontrolle sınanmamış dört modelde koşulması: Haiku 4.5, Sonnet 4.6, Opus 5.5, Fable 5.2 | `1fa4d54173f8e033ebf9985e5dd761407f0ac4e02f4f490351214997a1a8b280` |
| `results_c4_protocols.jsonl` | 1.200 | Üç protokol (standart, sınama yönergesi, yanıt denetimi) × dört koşul (kara-kutu, sade kod, zor koruyuculu kod, çürütülebilir kontrol) × dört model (Haiku 5.5, Sonnet 5.5, Gemini 3.5 Flash-Lite, Gemini 3.8 Flash) × 25 tohum | `150e70128d7c9e93174ed165f3a7d5d54cf7757ff6a652c6a10dde08cd72c5bf` |
| `results_c5_check_tools.jsonl` | 163 | Araç kullanımına izin veren yanıt denetimi × iki koşul (kara-kutu, zor koruyuculu kod) × 25 tohum. Haiku 5.5, Sonnet 5.5 ve Gemini 3.5 Flash-Lite tamdır (150 deneme). Gemini 3.8 Flash'ta sağlayıcıdaki ön ödemeli kredi tükendiği için 50 denemenin 13'ü vardır; bu 13 kayıt analize alınmaz. | `b0d80dab17b0638324327e15ca42adbce2e2a41c6dbd1aee97c780029eb59517` |

Sonnet 4.5 (`claude-sonnet-4-5-20250929`) bu tarihte API'de sunulmadığı için çürütülebilir kontrolle sınanamamıştır. Kayıtlardaki `protocol` alanı `verify_first` (sınama yönergesi), `harness_check` (yanıt denetimi) veya `harness_check_tools` (araç kullanımına izin veren yanıt denetimi) değerini alır; standart protokolde alan yoktur. Yanıt denetiminde `check_result` denetimin sonucunu (`pass`, `mismatch`, `invalid`), `first_pred_support` ve `first_pred_signs` ise denetimden önceki ilk yanıtı tutar.

`harness_check` protokolünde ajan dört tur kullanır ve denetim başarısız olursa araçsız tek bir düzeltme turu alır. `harness_check_tools` protokolünde ajan beş turunun tamamını kullanır; denetim başarısız olursa on denetim sorgusu `queries.json` dosyasına eklenir ve ajan üç tur daha alır (ilk ikisinde araç kullanabilir). Bu kayıtlarda `queries_before_check` ajanın denetimden önce harcadığı sorgu sayısıdır; `queries_used` denetimin on sorgusunu ve düzeltme turlarındaki sorguları da içerir.

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
| `pred_support`, `pred_signs`, `heldout_acc`, `matches_decoy` | Deney 2 tahmini, 100 test noktasındaki doğruluk ve tuzak kuralla eşleşme |
| `input_tokens`, `output_tokens`, `thinking_tokens`, `cost_usd`, `wall_s` | Token, maliyet ve süre ölçümleri |

## Makaledeki adlar ve kayıtlardaki etiketler

Makale, koşulları okunur Türkçe adlarla anar; kod ve ham kayıtlar ise aşağıdaki etiketleri kullanır.

| Makaledeki ad | Kayıt alanı | Etiket |
| --- | --- | --- |
| Zor rejim | `regime` | `vbb_hard` |
| Çözülebilir rejim | `regime` | `transparent` |
| Anahtarsız kontrol (sızıntı testi, η = 0,50) | `regime` | `control_impossible` |
| 4-XOR, n = 40 ve n = 80 (kalibrasyon aileleri) | `family`, `n` | `4xor` (`n` = 40 veya 80) |
| 3-XOR ailesi | `family` | `3xor` |
| Sertleştirilmiş 4-XOR ailesi | `family` | `4xor_strict` |
| Sabit boyutlu 4-XOR ailesi (n = 120) | `family` | `4xor_fixed120` |
| Sade kod (Deney 1) | `substrate` | `code_clean` |
| Karartılmış kod | `substrate` | `code_obf` |
| Seyrek ağırlık (iki katmanlı) | `substrate` | `relu_weights` |
| Yoğun ağırlık (üç katmanlı, ortogonal ön katman) | `substrate` | `relu_weights_dense` |
| Python araçlı / araçsız | `track` | `tool_python` / `tool_free` |
| Çoğunluk-3 görevi | `task_type` | `majority3` |
| Karar Listesi-3 görevi | `task_type` | `dlist3` |
| Sade kod (Deney 2) | `variant` | `wb_clean` |
| Kısa kontrol (kısa ölü-kod kontrolü) | `variant` | `wb_decoy_trivial_guard` |
| Çürütülebilir kontrol (bayt-özdeş) | `variant` | `wb_decoy_long_refutable_guard` |
| Zor koruyuculu kod | `variant` | `wb_inv_code` |
| Opak ağırlık | `variant` | `wb_inv_relu` |
| 3t/3t sıkı bütçe | `budget_track` | `matched_total_budget` |
| 5t/5t orta bütçe | `budget_track` | `matched_medium_budget` |
| 4t/8t asimetrik bütçe | `budget_track` | `matched_query_free_code` |
| 8t/8t eşit geniş bütçe | `budget_track` | `matched_wide_equal_budget` |
| Çıktı token tavanı 8.192 koşulu | `max_tokens_override` | `8192` |
| Sınama yönergesi protokolü (ek koşular) | `protocol` | `verify_first` |
| Yanıt denetimi protokolü (ek koşular) | `protocol` | `harness_check` |
| Araç kullanımına izin veren yanıt denetimi (ek koşular) | `protocol` | `harness_check_tools` |

Sekiz klasik saldırının kod adları (`scalar_b2`, `deg1`, `deg2_svd`, `deg2_bias_svd`, `probe_m3`, `canon_short_cover`, `canon_kikuchi`, `canon_sa`) `baselines.py` içindedir; ajana sunulan iki araç `runner.py` içinde `query_oracle` ve `run_python` adlarıyla tanımlıdır.

## Değerlendirilen modeller

- **Anthropic (9):** `claude-sonnet-4-5-20250929`, `claude-haiku-4-5-20251001`, `claude-opus-4-6`, `claude-sonnet-4-6`, `claude-opus-4-8`, `claude-haiku-5-5`, `claude-sonnet-5-5`, `claude-opus-5-5`, `claude-fable-5-2`
- **Google Gemini (4):** `gemini-3.1-flash-lite`, `gemini-3.5-flash-lite`, `gemini-3.8-flash`, `gemini-4-carbon`
- **Açık ağırlıklı Gemma 4 (2):** `gemma-4-26b-a4b-it`, `gemma-4-31b-it`

## Bilinmesi gerekenler

- `on_kayit.md` yerel bir protokol belgesidir; harici bir platformda zaman damgalı ön-kayıt yapılmamıştır. Belgedeki bazı bağlantılar yazarın yerel dosya yollarını gösterir ve depoda açılmaz.
- Ana Deney 1 ailesi olan `4xor_strict` parametreleri, kalibrasyon aşamasındaki bulgulardan sonra belirlenmiştir.
- Ek koşular sonuçlar görüldükten sonra tasarlanmıştır ve keşifseldir; makalede ana bulgulardan ayrı bir alt bölümde raporlanır.
- API sağlayıcıları model sürümlerini değiştirebildiği için canlı koşular kayıtlardaki sayıları birebir vermeyebilir. Kayıtlardan yapılan analiz ise deterministiktir.

## Atıf

Makale yayımlandığında atıf bilgisi buraya eklenecektir.
