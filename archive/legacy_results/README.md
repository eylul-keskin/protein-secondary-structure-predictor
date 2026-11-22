# Eski Sonuç Arşivi

Bu klasör, proje tesliminden kalan üç epokluk log, metrik ve eğitim geçmişi dosyalarını izlenebilirlik için saklar. Bu tablolar güncel checkpoint değerlendirmesinin sonuçları olarak kullanılmaz.

`run_summary.json` CPU üzerinde üç epokluk bir koşu kaydeder; `training_history.csv` üç satır içerir ve son doğrulama macro F1 değeri yaklaşık `0.285653` olur. İncelenen checkpoint ise eğitim yapılandırmasında altı epok sınırı ve kayıtlı en iyi doğrulama bilgisinde yaklaşık `0.263481` macro F1 taşır. Bu değer eski eğitim geçmişinin ilk satırıyla eşleşir; arşivdeki son skorun aynı ağırlık dosyasından geldiği doğrulanamamıştır.

Arşiv karışıklık matrislerinden türetilen metrikler JSON dosyalarıyla aritmetik olarak tutarlıdır. Bu tutarlılık, model ağırlıklarıyla tekrar üretilebilirlik kanıtı değildir. Bu klasörde model ağırlığı bulunmaz ve buradaki dosyalarla eşleşen bir ağırlık garantisi verilmez.

Dosyalar şu amaçlarla tutulur:

- `run_summary.json`: önceki koşunun kayıtlı ayarları ve sonuçları.
- `training_history.csv`: arşivlenmiş üç epokluk eğitim geçmişi.
- `*_metrics.json`, karşılaştırma, sınıf ve karışıklık CSV'leri: önceki skorların ve hataların incelenmesi.

Güncel sonuçlar, mevcut checkpoint kullanılarak bütün doğrulama ve test dosyalarında yeniden hesaplanmış ve proje kökündeki [`outputs/`](../../outputs/) klasörüne yazılmıştır. Bu hazırlıkta yeni eğitim yapılmamıştır. Güncel değerlendirme ile bu tarihsel arşiv birlikte tek bir eğitim koşusu gibi yorumlanmamalıdır.

Ayrıntılar: [deney denetimi](../../docs/EXPERIMENT_AUDIT.md).
