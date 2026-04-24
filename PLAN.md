# CSE476 LLM Projesi — Geliştirme Yol Haritası

## Context

Proposal: Scalable oversight / superalignment problemini araştır. Solib framework'ü üzerinde:
- LogiQA mantıksal akıl yürütme soruları
- LLM ile üretilmiş sentetik yanıltıcı (distractor) yanlış cevaplar
- Expert Debater (güçlü) vs Weak Judge (zayıf) asimetrik setup
- ASD (Agent Score Difference) metriği ile değerlendirme

Şu anki durum: Framework hazır ve çalışıyor, ama `init_exp.py` kırık (LogiQA loader yok), hiç deney çalıştırılmamış, results/ boş.

---

## Aşama 1 — Sistemin Çalıştığını Doğrula (Mevcut Dataset) ✅

**Hedef:** PrOntoQA ile küçük bir deney koşturup pipeline'ın uçtan uca çalıştığını gör.

**Değiştirilecek dosya:** `experiments/init_exp.py`

Yapılacaklar:
1. `LogiQA.data(limit=50)` → `PrOntoQA.data(limit=10)` olarak değiştir
2. `agent_models` listesini sadeleştir: `["claude-3-5-haiku-20241022"]`
3. `judge_models` listesini sadeleştir: `["claude-3-haiku-20240307"]`
4. `protocols = ["blind", "debate"]` (sadece 2 protocol)
5. `num_turns = [2]` (az tur)
6. `SIMULATE=True` ile cost estimation çalıştır → ardından `SIMULATE=False` ile gerçek deneyi koş

**Doğrulama:**
```bash
SIMULATE=True python experiments/init_exp.py
python experiments/init_exp.py
# experiments/results/ altında results.jsonl ve stats.json oluşmalı
```

---

## Aşama 2 — LogiQA Dataset Loader ✅ TAMAMLANDI

**Hedef:** `solib/data/loading.py`'a `LogiQA` sınıfı ekle.

**Dataset kaynağı:** HuggingFace `lucasmccabe/logiqa`.
- Format: çok şıklı mantık soruları (A/B/C/D), tek doğru cevap
- Mevcut pattern: `GSM8K`, `MMLU`, `PrOntoQA` sınıflarına bakarak aynı yapıyı uygula
- `extract_info()`: `question`, `answer_correct`, `answer_incorrect` ayıkla
- `to_question()`: `Question(question=..., answer_cases=[Answer(...), Answer(...)])` döndür

**Değiştirilecek dosya:** `solib/data/loading.py`

Yeni sınıf iskeleti:
```python
class LogiQA(Dataset):
    @classmethod
    def data(cls, limit=None, user_seed=0): ...
    def extract_info(self): ...  # correct + 1 random incorrect (binary)
    def to_question(self): ...
```

**Not:** Solib binary sorularla çalışıyor (2 cevap şıkkı). LogiQA 4 şıklı ama correct + en iyi distractor seçilerek binary'e indirgenecek.

---

## Aşama 3 — Sentetik Distractor Üretimi

**Hedef:** LogiQA'daki yanlış cevapları daha ikna edici hale getir ya da yeni yanıltıcı argümanlar üret.

**Yaklaşım:** Mevcut `solib/prompts/data_generation/` altındaki gsm8k_incorrect_*.jinja pattern'ine bakarak LogiQA için yeni Jinja2 şablonu yaz.

**Yeni dosyalar:**
- `solib/prompts/data_generation/logiqa_distractor_system.jinja`
- `solib/prompts/data_generation/logiqa_distractor_user.jinja`

**Üretim scripti:** `experiments/generate_logiqa_distractors.py`
- Girdi: LogiQA sorusu + doğru cevap
- Çıktı: `get_llm_response()` ile üretilmiş ikna edici ama yanlış distractor
- Sonuç: augmented JSON dosyası → `solib/data/logiqa/logiqa_augmented.json`

**Dataset loader güncelleme:** `LogiQA` sınıfına `augmented=True` parametresi ekle (sentetik vs orijinal distractor seçimi).

---

## Aşama 4 — Asimetrik Deney Konfigürasyonu

**Hedef:** Expert Debater (güçlü) vs Weak Judge (zayıf) setup.

**Değiştirilecek dosya:** `experiments/init_exp.py`

```python
questions = LogiQA.data(limit=50)

agent_models = [
    "claude-3-5-sonnet-20241022",  # Expert Debater (güçlü)
]
judge_models = [
    "claude-3-haiku-20240307",     # Weak Judge (zayıf)
]
protocols = ["blind", "debate", "consultancy"]
num_turns = [2, 4]
bon_ns = [1]
debate_toggle = [True, False]      # simultaneous vs sequential
```

**Baseline:** Blind protokol ile judge'ın zero-shot başarısını ölç → bu "incompetence baseline".

---

## Aşama 5 — Analiz ve Görselleştirme

**Hedef:** ASD metriğini çiz, protokol karşılaştırması yap.

**Yeni dosya:** `experiments/logiqa_analysis.py`

Üretilecek grafikler:
1. **ASD per protocol**: Blind / Propaganda / Debate / Consultancy karşılaştırması
2. **Judge accuracy trajectory**: Tur sayısına göre doğruluk değişimi
3. **Sycophancy indicator**: Güçlü debater'ın zayıf judge'ı ne kadar etkilediği

**Mevcut kullanılabilir:** `solib/analysis.py`, `experiments/init_exp_analyze.py`, `experiments/analysis/` altındaki JSON'lar referans alınabilir.

---

## Aşama 6 — Lokal Model Setup (Sonraki Adım)

**Hedef:** Ollama üzerinde open-weights modeller (Llama3, Mistral).

- `.env`'de `SIMULATE=False` ve Ollama endpoint konfigürasyonu
- `llm_utils.py`'daki `ollama_chat/` prefix desteği zaten mevcut
- `judge_models = ["ollama_chat/llama3:8b"]` gibi konfigürasyon
- Lokal model = Weak Judge, API model = Expert Debater (asimetrik)

---

## Kritik Dosyalar

| Dosya | Aşama |
|-------|-------|
| `experiments/init_exp.py` | 1, 4 |
| `solib/data/loading.py` | 2 |
| `solib/prompts/data_generation/logiqa_*.jinja` | 3 (yeni) |
| `experiments/generate_logiqa_distractors.py` | 3 (yeni) |
| `experiments/logiqa_analysis.py` | 5 (yeni) |

## Yeniden Kullanılacak Mevcut Fonksiyonlar

- `solib/utils/llm_utils.py` → `get_llm_response()` (distractor üretimi için)
- `solib/utils/random()` → reproducibility için
- `solib/data/loading.py` → `GSM8K`, `MMLU` sınıfları (LogiQA için şablon)
- `solib/prompts/data_generation/gsm8k_incorrect_*.jinja` → distractor prompt şablonu
- `experiments/init_exp_analyze.py` → analiz için referans
- `solib/Experiment.py` → `Experiment.recompute_stats()` (sonuçlar üzerinde re-analiz)

## Doğrulama

```bash
# Aşama 1
uv run pytest -s                              # testler geçmeli
SIMULATE=True python experiments/init_exp.py  # maliyet tahmini
python experiments/init_exp.py                # gerçek deney

# Aşama 2-3
python -c "from solib.data.loading import LogiQA; print(LogiQA.data(limit=5))"

# Aşama 4
python experiments/init_exp.py                # LogiQA + asimetrik setup

# Aşama 5
python experiments/logiqa_analysis.py         # grafikler üretilmeli
```
