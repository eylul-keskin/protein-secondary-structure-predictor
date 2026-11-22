# Deney kaydı denetimi ve yeniden değerlendirme

İnceleme tarihi: 19 Kasım 2026. Bu belge, özgün teslimdeki eğitim arşivi ile bu yayında yeniden hesaplanan değerlendirmeyi ayırır.

## Bulunan tutarsızlık

Özgün `run_summary.json` üç epokluk CPU koşusu kaydederken checkpoint yapılandırması `epochs=6` içeriyordu. Ayrıca checkpoint'in doğrulama metadatası, eski eğitim geçmişinin ilk satırıyla eşleşiyordu; eski sonuç JSON'ları üçüncü epok değerlerini içeriyordu.

| Doğrulama ölçüsü | Checkpoint metadatası / eski geçmişin 1. epoku | Eski sonuç JSON'u / 3. epok |
| --- | ---: | ---: |
| Accuracy | 0,5347646297 | 0,5745810522 |
| Macro F1 | 0,2634813437 | 0,2856530297 |
| Weighted F1 | 0,4932848943 | 0,5302224277 |
| Loss | 1,2832516986 | 1,1921457319 |

Bu, dosyaların aynı eğitimin doğrulanmış son sürümü olarak birlikte sunulamayacağını gösterir. Dosyaların ne zaman veya nasıl ayrıştığı belirlenmemiştir. Checkpoint'teki `epochs=6` bir yapılandırma sınırıdır; gerçekten tamamlanan epok sayısı bilinmiyor olarak bırakılmıştır. Bu değeri üçe çevirerek veya eski skorları koruyarak tutarsızlık gizlenmemiştir.

Eski günlük, eğitim geçmişi ve metrikler [`archive/legacy_results/`](../archive/legacy_results/) altında açıklamalı olarak korunur. Ana `outputs/` dizini yeni değerlendirmeyi içerir; eski eğitim eğrisi bu modelin eğitim kanıtı olarak kullanılmaz.

## Ne yeniden çalıştırıldı?

Model ağırlıkları değiştirilmeden, `weights_only=True` ile yüklenen aynı checkpoint kullanıldı. Eğitim CSV'sinden sözlük ve çoğunluk etiketi tekrar kuruldu; sözlük/etiket haritaları kayıtlı bilgilerle karşılaştırıldı. Referans ve model, validation ile üç test kümesinde değerlendirildi. Optimizasyon adımı veya yeniden eğitim yapılmadı.

Yeniden değerlendirme için veri dosyaları ve uyumlu bir yerel checkpoint gerekir. Veri dosyalarının düzeni [veri notlarında](DATA_SOURCES.md) yer alır.

Taşınabilir [`reevaluate_checkpoint.py`](../reevaluate_checkpoint.py) betiği:

- Var olan çıktı dizinini reddeder ve yeni bir dizine yazar.
- CPU'da iki iş parçacığı ve batch size 16 kullanır.
- Veri, kaynak kod ve checkpoint hash'lerini kaydeder; veri ve ağırlıkların işlem sonunda değişmediğini kontrol eder.
- Tamamlanmış değerlendirme için `status: completed` kaydeder.
- Metrik, sınıf raporu, karışıklık matrisi, uzunluk analizi ve örnek tahminleri üretir; eğitim geçmişi üretmez.

Çalıştırma, repo kökünde yeni bir çıktı dizini belirtilerek yapılabilir:

```shell
python -B reevaluate_checkpoint.py --output-dir reevaluation
```

Checkpoint SHA-256:

```text
5a9fc61372715f10b0e97eb420052096f3563efa1994430ecb58ac6e448acbb0
```

## Tutarlılık kontrolünün sonucu

Yeni doğrulama accuracy değeri checkpoint metadatasıyla birebir eşleşti. Macro F1 farkı yaklaşık `2,0128e-7`, weighted F1 farkı `1,9794e-7`, loss farkı `1,5824e-9` oldu. Kullanılan açık mutlak tolerans, sınıflandırma ölçülerinde `1e-6`, loss için `1e-5` idi. Küçük farkların nedeni ayrıca kanıtlanmadı; güncel CSV ve JSON'lar fiilen yeniden hesaplanan değerlerdir.

İlk denetim betiğindeki `1e-10` eşiği bu küçük farkta durdu. Eşik açıkça düzeltildi; üretilmiş doğrulama tahminleri tekrar çalıştırılmadan, hash'leri kontrol edilerek kalan üç test tamamlandı. İki aşama ve tam farklar [`evaluation_summary.json`](../outputs/logs/evaluation_summary.json) içinde; ilk duruş [`validation_strict_tolerance_stop.json`](../outputs/logs/validation_strict_tolerance_stop.json) içinde kayıtlıdır. Her değerlendirme kümesinin model tahminleri bir kez hesaplandı.

Yeni sonuçlar dört dosyada da çoğunluk referansını aşar; sınıf bazlı ciddi eksikler devam eder. Güncel skor ve yorumlar [proje raporundadır](PROJECT_REPORT.md).

## Doğrulanan ve doğrulanmayan kapsam

- On iki özgün Python dosyası ve beş CSV içerik olarak korunmuştur. Eğitim/model matematiği değiştirilmemiştir; yeni değerlendirme betiği eklenmiştir.
- İki gerçek doğrulama proteiniyle ilk kontrolde 636/184 uzunluk, 452 dolgu konumu ve sonlu kayıp/logitler doğrulandı. Tekli ve dolgulu batch çıktılarının en büyük farkı `4,77e-7` idi.
- Tam değerlendirmede toplam accuracy/F1/loss değerlerinin sonluluğu kontrol edildi. Her ara tensör için ayrı bir kapsamlı denetim yapılmış değildir.
- Eski sekiz karışıklık matrisinden hesaplanan metrikler eski JSON'larla `1e-12` toleransında tutarlıydı; bu aritmetik uyum, eski checkpoint bağlantısını doğrulamıyordu.
- İki notebook'un eğitim geçmişi bağımlılıkları ve güncelliğini yitirmiş açıklamaları düzeltildi; kod hücreleri sözdizimi kontrolünden geçti. Notebook'lar baştan sona yürütülmüş çıktı kanıtı olarak sunulmaz.
- Özgün teslim PDF'si ve makineye özgü sanal ortamlar bu pakete alınmadı. Özgün dosyalar yerelde korunur. Mevcut veri kaynakları ayrıca [DATA_SOURCES.md](DATA_SOURCES.md) içinde açıklanır.

Kontrol ortamı Python 3.12.10, PyTorch 2.12.1+cu130, pandas 3.0.3, NumPy 2.4.4, scikit-learn 1.8.0 ve Matplotlib 3.10.9 idi; değerlendirme CPU'da yapıldı. Bu sürümlerin bir bölümü özgün `requirements.txt` sabitlemelerinden farklıdır. Temiz ortamda bu gereksinimlerin kurulumu veya önceki eğitimin baştan üretimi bu incelemede yapılmadı. Donanım ve sürüm değişikliklerinde birebir aynı sonucu garanti etmek yerine ortam, hash ve tolerans kaydı sunulur.
