# Ön-Kayıt Protokolü (Pre-Registration) — Pilot Deneyi
**Çalışma:** Gözlemciye Göre Sanal Kara-Kutu (VBB) Kuramı, Tek-Ajan Hesaplamalı Blackwell Tersinmesi ($\Delta_{\mathrm{WB}} < 0$) ve Ayrık-Girdili Düşük-Dereceli Nöral-Cebirsel Derleme  
**Tarih:** 2026-11-21 (Herhangi bir ücretli API çağrısı yapılmadan önce kilitlenmiştir)  
**Kod Tabanı:** [pilot/](file:///home/user/Belgeler/benchmark/pilot) (`generators.py`, `validators.py`, `baselines.py`, `sandbox.py`, `runner.py`, `analyze.py`)

---

## 1. Amaç ve Sınanabilir Önermeler (H1–H4)

Bu pilot deney, [bulus_asama_b_kuram.md](file:///home/user/Belgeler/benchmark/bulus_asama_b_kuram.md) belgesinde biçimselleştirilen iki teoremin (Teorem 1: Hesaplamalı Blackwell Tersinmesi; Teorem 2: Beyaz-Kutu Denetim $\equiv$ *Planted Noisy $k$-XOR*) Anthropic Claude modelleri üzerinde **düşük bütçeyle (\$25–\$45) yanlışlanabilir biçimde sınanması** için tasarlanmıştır.

- **Ö1 / H1 (Ortalama-Durum Nöral-Cebirsel VBB Bariyeri):** `4-XOR` ($n=40, \eta=0{,}10$) ailesinde:
  - **VBB-zor rejimde** ($m=180$, $\log_2 \mathrm{FP\_UB} = -27{,}9$, bilgi-kuramsal avantaj $\approx 1{,}00$), ne kara-kutu (`BB`) ne de Python-araçlı beyaz-kutu (`WB`) Claude modelleri arka-kapılı $\mathcal{N}_P$ ile temiz $\mathcal{N}_Q$ ağlarını şansın istatistiksel olarak üstünde ayırt edebilir ($\widehat{\mathrm{Adv}} < 0{,}15$ veya Holm-düzeltmeli $p_{\mathrm{Holm}} \ge 0{,}05$).
  - **Şeffaf rejimde** ($m=550$, `Canon+Kikuchi-2` avantajı $= +1{,}00$), en az bir güçlü araçlı Claude ajanı (`Sonnet` veya `Opus` katmanı) beyaz-kutu ortamında anlamlı pozitif beyaz-kutu avantajına geçer ($\widehat{\Delta}_{\mathrm{WB}} \ge +0{,}30$ ve $p_{\mathrm{Holm}} < 0{,}05$).
- **Ö2 / H2 (Tek-Ajan Hesaplamalı Blackwell Tersinmesi — Ana Deneysel Keşif):** Davranışı %100 koruyan $\mathcal{C}_{\mathrm{inv}}$ derleyicisi (`wb_inv_code` ve birleşik 3-gizli-katmanlı ReLU ağı `wb_inv_relu`) üzerinde, `f_true` (3-değişkenli çoğunluk kuralı, $n=40$) görevi için eşleştirilmiş tohumlarda (`seed`-paired):
  - **Şeffaf kontrol (`wb_clean`):** $\widehat{\Delta}_{\mathrm{WB}} \ge 0$ (beyaz-kutu temiz kodda kara-kutuya eşit veya üstün).
  - **Mekanizma I + II (*Eşleştirilmiş Toplam Bütçe* izi, `matched_total_budget`):** Aynı tur (`max_turns=3`), aynı token (`max_output_tokens=1800`) ve aynı kahin kotasında (`q_max=80`), `wb_inv_code` veya `wb_inv_relu` üzerinde $\widehat{\Delta}_{\mathrm{WB}} \le -0{,}05$ ve eşleştirilmiş işaret-çevirme permütasyon testi $p_{\mathrm{Holm}} < 0{,}05$ ($\hat{\Gamma}_{\mathrm{WB}} < 0$).
  - **Saf Mekanizma II (*Eşleştirilmiş Sorgu + Ücretsiz Kod* izi, `matched_query_free_code`):** `WB` kipine aynı `q_max=80` sorgu kotasının **yanı sıra kesin olarak daha fazla tur (`8` vs `4`) ve daha fazla çıktı tokenı (`4096` vs `2048`)** verilmesine rağmen, $\widehat{\Delta}_{\mathrm{WB}} \le -0{,}05$ ve $p_{\mathrm{Holm}} < 0{,}05$ gözlenmesi saf epistemik/önsel tersinmesini (*"sadece bütçe yetmedi"* itirazını kapatarak) doğrular.
- **Ö3 / H3 (Üç Ortamlı Temsil Hiyerarşisi):** Aynı mantıksal işlev $f$ için şeffaf rejimde $\widehat{\Delta}_{\mathrm{WB}}(\text{code\_clean}) \ge \widehat{\Delta}_{\mathrm{WB}}(\text{code\_obf}) \ge \widehat{\Delta}_{\mathrm{WB}}(\text{relu\_weights})$ sırası ve `code_clean` ile `relu_weights` arasında ölçülebilir fark gözlenir.
- **Ö4 / H4 (Katman / Düşünme Bütçesi Ölçeklenmesi):** Model yeteneği/düşünme bütçesi arttıkça (`Haiku` $\to$ `Sonnet` $\to$ `Opus` / extended thinking), şeffaf rejimde $\Delta_{\mathrm{WB}}$ artarken, $\mathcal{C}_{\mathrm{inv}}$ üzerinde $|\Delta_{\mathrm{WB}}|$'nin küçülüp küçülmediği (tersine çevirmeyi aşma vs. *aşırı statik analizle* daha derin tersinme) yön ön-varsayımı olmaksızın iki-yönlü ölçülür.

---

## 2. Model Seçimi ve Hesap Doğrulaması

Gerçek API çalıştırmasından hemen önce:
```bash
python3 -m pilot.runner --list-models
```
komutu çalıştırılarak kullanıcının Anthropic hesabında aktif olan kesin model kimlikleri doğrulanacaktır. Varsayılan üç katmanlı pilot hedefi (hesapta aktif sürümlere göre kesinleştirilecek):
1. **Hızlı / Küçük katman:** `claude-haiku-5-5` (veya aktif en güncel Haiku)
2. **Orta / Ana ajan katmanı:** `claude-sonnet-5-5` (veya aktif en güncel Sonnet)
3. **Üst / Derin akıl yürütme katmanı (seçili kritik hücrelerde):** `claude-opus-5-5` (veya aktif en güncel Opus; ayrıca `Sonnet` üzerinde `--thinking-budget 2048` karşılaştırması)

---

## 3. Pilot Deney Izgarası, İstatistiksel Güç ve Bütçe Planı

### 3.1 Deney 1 (`exp1_vbb` — `4-XOR`, $n=40$, `tool_python`)
- **Rejimler (4):** `control_impossible` ($\eta=0{,}50$, $m=180$ — sızıntı/ezberleme kapısı), `vbb_hard` ($m=180$, $\eta=0{,}10$), `transition` ($m=320$, $\eta=0{,}10$), `transparent` ($m=550$, $\eta=0{,}10$).
- **Erişim & Ortamlar (4):** `BB` + `WB(code_clean, code_obf, relu_weights)`.
- **Örneklem büyüklüğü:** Her hücrede $20$ çift = **$N = 40$ örnek** (20 `PLANTED` + 20 `CLEAN`); ikili avantaj standart hatası $\mathrm{SE}(\widehat{\mathrm{Adv}}) \le 1/\sqrt{N} = 0{,}158$ (şeffaf rejimde $\mathrm{Adv} \ge 0{,}50$ için tek-yönlü Fisher gücü $> \%95$).
- **Hücre sayısı:** 2 model (`Haiku`, `Sonnet`) $\times$ 4 rejim $\times$ 4 kip/ortam $\times$ 40 örnek = **1.280 deneme** (ek olarak `Opus` yalnızca `vbb_hard` + `transparent` $\times$ `code_clean, relu_weights` = 160 deneme).

### 3.2 Deney 2 (`exp2_blackwell` — `C_inv`, $n=40$, $|S^*|=3$, `q_max=80`, `tool_python`)
- **Varyantlar (3):** `wb_clean` (şeffaf kontrol), `wb_inv_code` (MBA koruyucu + ortogonal ikiz-ReLU saklı `f_true` + belirgin tuzak), `wb_inv_relu` (birleşik 3-gizli-katmanlı, sıfır-ipuçlu ReLU MLP).
- **Bütçe İzleri (2):** `matched_total_budget` (Mekanizma I+II) ve `matched_query_free_code` (Saf Mekanizma II).
- **Örneklem büyüklüğü:** Her hücrede **$N = 40$ eşleştirilmiş tohum** (`seed` 5000..5039). Eşleştirilmiş doğruluk farkında $\sigma_{\Delta} \approx 0{,}15$ varsayımıyla $\delta = 0{,}08$ etkisi için eşleştirilmiş test gücü $> \%90$; $\delta = 0{,}15$ için $> \%99$.
- **Hücre sayısı:** 2 model (`Haiku`, `Sonnet`) $\times$ 2 bütçe izi $\times$ 4 kip (`BB`, `wb_clean`, `wb_inv_code`, `wb_inv_relu`) $\times$ 40 tohum = **640 deneme** (ek olarak `Opus` üzerinde `matched_total_budget` + `matched_query_free_code` $\times$ `BB, wb_inv_code, wb_inv_relu` $\times$ 30 tohum = 180 deneme).

### 3.3 Tahmini Maliyet ve Sert Harcama Sınırı
| Paket | Tahmini Çağrı | Tahmini Maliyet (USD) |
| :--- | :---: | :---: |
| Exp 1 (`Haiku` + `Sonnet`, 1.280 deneme) | ~2.600 mesaj | ~\$10–\$16 |
| Exp 2 (`Haiku` + `Sonnet`, 640 deneme) | ~1.500 mesaj | ~\$7–\$12 |
| `Opus` kritik doğrulama dilimi (340 deneme) | ~800 mesaj | ~\$8–\$14 |
| **Toplam Pilot (Sert Tavan `--max-cost-usd 45.0`)** | **~4.900 mesaj** | **~\$25–\$42** |

*Not:* Tüm çalıştırmalar `pilot_results.jsonl` dosyasına adım-adım kaydedilir ve kaldığı yerden devam eder (*resumable*). Önce `Haiku` ile ucuz ön-tarama (\$2–\$4) yapılıp sızıntı kapısı doğrulandıktan sonra `Sonnet` ve `Opus` dilimlerine geçilecektir.

---

## 4. Kalite, Güvenlik ve Sızıntı Kontrol Kapıları

1. **Sızıntı / Ezberleme Kapısı (`control_impossible`, $\eta=0{,}50$):** Bu hücrelerde `PLANTED` ile `CLEAN` dağılımları matematiksel olarak **özdeştir** ($P \equiv Q$). Herhangi bir model/ortamda $|\widehat{\mathrm{Adv}}| \ge 0{,}10$ **ve** $p_{\mathrm{Holm}} < 0{,}05$ gözlenirse deney derhal **GEÇERSİZ (`INVALID_LEAK_DETECTED`)** sayılır ve kod/veri hattındaki sızıntı giderilmeden sonuç raporlanmaz.
2. **Ayrıştırma (*Parse*) Başarısı Kapısı:** Her hücrede geçerli JSON yanıt oranı `< %90` ise ilgili hücre ayrı raporlanır; ayrıştırılamayan yanıtlar gizlice silinmez (`Exp 1`: yanlış, `Exp 2`: `heldout_acc = 0.50` olarak puanlanır).
3. **Kahin ve Sandbox Güvenliği:** `MeteredOracle` yalnızca üst süreçte yaşar; `IsolatedPythonSandbox` alt sürecine `ANTHROPIC_API_KEY` ve hiçbir gizli değişken geçirilmez (`test_pilot.py` içinde birim testiyle doğrulanmıştır).
4. **Çoklu Test Düzeltmesi:** Tüm birincil çıkarımlar aile-bazlı hata oranını ($\mathrm{FWER} \le 0{,}05$) koruyan **Holm–Bonferroni** adım-aşağı düzeltmesi ile raporlanır.

---

## 5. Aşama D Karar Kuralları (Ön-Kayıtlı — Değiştirilmez)

Pilot verisi toplandıktan sonra [analyze.py](file:///home/user/Belgeler/benchmark/pilot/analyze.py) çıktısına göre üç olası karardan biri mekanik olarak seçilir:

1. **TAM ÇALIŞMAYA GEÇ (Top-Venue Hedefi: NeurIPS / ICML / ICLR / IEEE S&P):**
   - Sızıntı kapısı **PASS**, **ve**
   - **H1** `SUPPORTED` (`vbb_hard` bariyeri korunmuş, `transparent` çözülmüş) **ve** **H2** en azından `SUPPORTED_MECH1_ONLY` (tercihen `SUPPORTED_BOTH_MECH1_AND_MECH2`, $\widehat{\Delta}_{\mathrm{WB}} \le -0{,}08$, $p_{\mathrm{Holm}} < 0{,}05$, $\hat{\Gamma}_{\mathrm{WB}} < 0$), **ve** `wb_clean` kontrolünde $\widehat{\Delta}_{\mathrm{WB}} \ge -0{,}02$.
   - *Eylem:* 3-XOR ($n=80$), yoğun ortogonal katman (`relu_weights_dense`), daha fazla model nesli/ailesi ve $N=100$/hücre tam ölçeğe geçilir.
2. **KAPSAMI DARALT / YENİDEN KALİBRE ET:**
   - Sızıntı kapısı **PASS**, fakat ya (a) H1'de hiçbir model `transparent` rejimini çözememiş (`INCONCLUSIVE_TRANSPARENT_UNSOLVED` — Kikuchi/kısa-örtü ipucu veya $m$ artışı gerekir), ya da (b) H2 yalnızca Mekanizma I'de çıkmış ama Mekanizma II'de ajanlar `artifact.py`'yi Python'da doğrudan örnekleme ile çözerek tersinmeyi aşmış.
   - *Eylem:* Zayıf kanat düşürülür veya yalnızca çalışan mekanizma + ek model/ortam ayrıştırmasıyla daraltılmış kuramsal-deneysel makale planlanır.
3. **FİKRİ BIRAK / KÖKTEN DEĞİŞTİR:**
   - H1'de `vbb_hard` rejiminde beklenmedik bir kırılma (`REFUTED_VBB_BREACHED` — inşada kusur!) **veya** H2'de hem Mekanizma I hem Mekanizma II'de hiçbir modelde anlamlı $\Delta_{\mathrm{WB}} < 0$ gözlenmemesi (`NOT_SUPPORTED` — tüm modeller $\mathcal{C}_{\mathrm{inv}}$'yi sorunsuz aşıyor).
   - *Eylem:* Hata analizi yazılır ve bu ampirik çizgi terk edilir.

---

## 6. Ön-Kayıt Değişikliği (Amendment v1.1 — 2026-11-21, Aşama C.1 Bulguları Sonrası)

Aşama C.1 (`claude-haiku-5-5`, 680 canlı deneme, `\$2.39`, [report_c1_haiku55.md](file:///home/user/Belgeler/benchmark/pilot/report_c1_haiku55.md)) sızıntı kapısını (`PASS`) geçmiş, H2 Mekanizma I (`Delta_WB = -0.199, p_holm = 0.0112`) ve H3'ü (`code_clean > code_obf > relu_weights`) doğrulamış, ancak iki kritik metodolojik nokta ortaya çıkarmıştır. Aşama C.2 başlamadan önce yapılan değişiklikler şeffaf biçimde kayda geçirilmiştir:

1. **Deney 1 (`4-XOR`) SA + GE + Kikuchi Sertleştirmesi:**
   - Eski $(n=40, m=180, \eta=0{,}10)$ noktası `claude-haiku-5-5` tarafından Python'da **Simulated Annealing** ile kırılmıştır (`Adv = +1.000`, sonlu-$n$ spin-cam yerel araması eşiği aştı).
   - Yeni kilitli `4-XOR` rejimi: **$n = 80, \eta = 0{,}15$** — `control_impossible` ($m=400, \eta=0{,}50$), `vbb_hard` ($m=400$, $\log_2 \mathrm{FP\_UB} = -35{,}8$, $\mathrm{GE} = 2{,}3\times 10^{-6}$, 180.000 adımlı SA ve `Kikuchi-2` kör), `transition` ($m=1400$), `transparent` ($m=2200$, `Canon+Kikuchi-2` $= +1{,}00$, ağır SA hâlâ eşiğe ulaşamaz). Eski rejim `4xor_n40` adıyla saklanmıştır.
   - `baselines.py`'ye **Simulated Annealing** 8. referans kırıcı olarak eklenmiştir.
2. **Düşünme Modu (`thinking: {"type": "adaptive"}`) ve Efor Seviyesi (`output_config.effort`) Kontrolü:**
   - Canlı API şema doğrulaması, Claude 5.5 ailesinde (`haiku-5-5`, `sonnet-5-5`, `opus-5-5`) adaptif düşünmenin varsayılan açık olduğunu, `claude-sonnet-4-6`'da ise `thinking: {"type": "adaptive"}` gönderildiğinde 4 modelin tamamının birebir aynı şemayı ve `output_config.effort` (`low`, `medium`, `high`) parametresini desteklediğini doğruladı.
   - Aşama C.1'de `BB` kipinde `max_tokens = 1800` sınırının adaptif düşünme bloklarıyla dolup 6 denemeyi yarım kestiği saptandığından, tüm Aşama C.2 çalıştırmalarında `max_tokens = 4096` (`effort=high` için `8192`) olarak sabitlenmiş; `thinking_tokens` ve `truncated_turns` her denemede ayrı loglanmaya başlanmıştır.
3. **Bütçe-Faz Eğrisi ve Efor Nedensel Ablasyonu (H4):**
   - Deney 2'de üç bütçe izi ölçümlenir: `matched_total_budget` (3 tur), `matched_medium_budget` (5 tur) ve `matched_query_free_code` (`BB=4, WB=8`).
   - Aynı model üzerinde `effort=low` vs `effort=medium` vs `effort=high` karşılaştırması ile düşünme eforunun $\Delta_{\mathrm{WB}}$ üzerindeki nedensel etkisi doğrudan ölçülür.
