# Protein SST8: hedef, deney ve eleştirel değerlendirme

## Ne amaçlandı?

Bu ders projesi, protein aminoasit dizisindeki her konum için sekiz ikincil yapı etiketinden birini tahmin eder. Girdi `seq`, hedef `sst8` sütunudur. `sst3` dosyalarda bulunur; özellik veya eğitim hedefi olarak kullanılmaz.

Çalışmanın soruları: Öğrenilen bir dizi modeli, her konumda en sık etiketi söyleyen referansı aşabiliyor mu? Ortalama skor hangi sınıflardaki başarısızlığı gizliyor? Verilen değerlendirme dosyalarında sonuç nasıl değişiyor? Amaç, bu soruları veri denetimi, doğru maskeleme ve hata analiziyle ele almaktır.

Kapsam, bir çoğunluk-sınıfı referansı ve tek bir Embedding + BiLSTM modelidir. Bu bir üç boyutlu protein yapı çözümü, yeni mimari veya alanın en iyi yöntemi iddiası değildir.

## Ne yapıldı?

Beş CSV; sütun, boş kayıt, dizi–etiket hizalaması, etiket kümesi, tekrar ve tam dizi örtüşmeleri açısından incelenir. Aminoasit sözlüğü ve etiket kodlayıcı yalnız eğitim verisinden kurulur. `<PAD>` ve `<UNK>` dahil 23 sözlük girdisi, sekiz hedef etiketi vardır.

Referans, eğitim kalıntılarında en sık görülen `H` etiketini her konum için tahmin eder; aminoasit içeriğini kullanmaz. Ana model öğrenilen aminoasit temsillerini çift yönlü LSTM'den geçirir ve her konumu doğrusal bir katmanla sınıflandırır. Modelde 202.184 öğrenilebilir parametre bulunur.

Dolgu etiketleri `-100` ile kayıptan çıkarılır. `pack_padded_sequence` gerçek uzunlukları kullanır; tahminler de gerçek uzunlukta çözümlenir. Dolgu konumları doğruluk ve F1'e katılmaz.

Eğitim kodu, her epokta doğrulama macro F1'e göre kontrol noktası seçer; test dosyaları bu seçim döngüsüne girmez. **Bu yayın hazırlığında eğitim yeniden çalıştırılmamıştır.** Arşiv değerlendirmesinde kullanılan ağırlıklar; doğrulama, CB513, TS115 ve CASP12 üzerinde yeniden değerlendirilmiştir. Eğitim geçmişindeki tutarsızlık ve yapılan denetim [deney kaydı raporunda](EXPERIMENT_AUDIT.md) açıklanır.

## Parametre ve hiperparametreler

Modelin öğrendiği ağırlıklar parametredir. Mimari ve eğitim seçimleri hiperparametredir. Aşağıdakiler checkpoint'e yazılmış ayarlardır; sistematik aramayla optimum değerler olarak seçildiklerine dair kayıt yoktur.

| Ayar | Değer | Ne belirler? |
| --- | ---: | --- |
| Embedding boyutu | 64 | Her aminoasidin öğrenilen temsil boyutu. |
| Gizli boyut | Her yön için 128 | İki yön birleşince sınıflandırıcıya 256 özellik verilir. |
| LSTM katmanı | 1 | Tek çift yönlü katman. |
| Dropout | 0,3 | Eğitimde embedding ve LSTM çıktılarında uygulanır. Tek katmanda LSTM'nin katmanlar arası dropout'u sıfırdır. |
| Batch size | 16 | Birlikte işlenen dizi sayısı. |
| Optimizer / öğrenme oranı | Adam / 0,001 | Güncelleme yöntemi ve adım ölçeği. |
| Weight decay | 0,00001 | Ağırlık büyüklüğüne düzenleme terimi. |
| Gradient clipping | 1,0 | Gradyan normunu sınırlar. |
| Patience | 3 | İyileşmeyen epoklar için erken durdurma eşiği. |
| Seed | 42 | Kayıtlı rastgelelik tohumu. |
| Yapılandırılmış epok sınırı | 6 | Tamamlanan epok sayısının kanıtı değildir; bu ağırlıkların gerçek eğitim süresi doğrulanamamıştır. |

Yeniden değerlendirme CPU, iki iş parçacığı ve batch size 16 ile yapılmıştır. Bu işlem ağırlıkları güncellemez.

## Güncel ölçümler

Tablo, önceki checkpoint ile yeniden üretilen değerlendirmeyi gösterir. Eski üç epokluk sonuç tablosu güncel modelin sonucu olarak kullanılmaz.

| Bölme | Referans accuracy | BiLSTM accuracy | Referans macro F1 | BiLSTM macro F1 |
| --- | ---: | ---: | ---: | ---: |
| Validation | 0,3303 | 0,5348 | 0,0621 | 0,2635 |
| CB513 | 0,2991 | 0,5052 | 0,0576 | 0,2560 |
| TS115 | 0,3513 | 0,5443 | 0,0650 | 0,2638 |
| CASP12 | 0,2741 | 0,4943 | 0,0538 | 0,2464 |

BiLSTM, bu kurulumda sabit-sınıf referansını dört dosyada da aşar. Referans çok basittir; aminoasit başına koşullu bir model, tek yönlü LSTM veya başka mimariyle kontrollü karşılaştırma yoktur. Kazanımın ne kadarının bağlamdan, çift yönlülükten veya diğer bileşenlerden geldiği ayrıştırılamaz.

**Accuracy**, doğru etiketlenen kalıntıların oranıdır. **Macro F1**, sekiz sınıfın F1 değerlerini eşit ağırlıkla ortalar. **Weighted F1**, sınıfları gerçek örnek sayılarına göre ağırlıklandırır. Kalıntılar proteinlerden birlikte toplandığından bunlar protein başına eşit ağırlıklı ölçüler değildir; uzun proteinler toplam ölçüme daha çok katkı verir.

### Ortalama skorun gizlediği durum

Doğrulamada `H`, `E` ve `C` F1 değerleri yaklaşık 0,6745; 0,5567 ve 0,5119'dur. `B` ve `I` ise **hiç tahmin edilmemiştir**: ilgili tahmin sütunları, recall ve F1 sıfırdır. `G` recall yaklaşık %0,34, `S` recall %1,62'dir. Dolayısıyla %53,48 doğruluk, sekiz sınıfın tamamının kullanışlı biçimde öğrenildiğini göstermez.

Eğitim verisinde `H` yaklaşık %31,85; `B` %0,95 ve `I` %0,50 oranındadır. Sonuçlar dengesizlikle uyumludur; fakat sınıf ağırlığı veya alternatif kayıp deneyi bulunmadığından dengesizlik tek neden olarak gösterilemez.

## Güçlü yönler ve sınırlar

| Güçlü taraf | Sınırı |
| --- | --- |
| Eğitimden kurulan sözlük, hizalama ve açık dolgu maskesi | Bunlar değerlendirme kümelerinin bağımsızlığını garanti etmez. |
| Doğrulama macro F1 ile model seçen eğitim kodu | Tek ayar kümesi ve seed vardır; belirsizlik aralığı veya optimumluk kanıtı yoktur. |
| Sınıf raporları ve karışıklık matrisleri | Bazı sınıflarda açık başarısızlık vardır; ortalama skor tek başına yeterli değildir. |
| Üç ayrı test dosyası | Aralarında ortak diziler bulunur; bağımsız tekrarlar olarak yorumlanamaz. |
| Uzunluk grupları ve örnek tahminler | Analiz betimseldir; örnekler rastgele değil, hata/başarı ve uzunluk uçlarından seçilir. |
| Model/veri hash'leriyle yeni değerlendirme kaydı | Önceki eğitimin bütün geçmişi geri kazanılmış değildir. |

CB513'te 79 fazla tekrar satırı vardır. Eğitim–CB513 arasında iki, eğitim–TS115 arasında bir ortak tam dizi bulunur. Eğitimle ortak kalıntı oranı sırasıyla yaklaşık %0,063 ve %0,912'dir; bunların bütün kazanımı açıkladığı söylenemez. TS115 ile CASP12 dört diziyi paylaşır; ortak 982 kalıntı küçük CASP12 dosyasının yaklaşık %13,53'üdür. Homoloji veya protein ailesine göre bağımsızlık denetimi yapılmamıştır. Ayrıntılar [veri notlarındadır](DATA_SOURCES.md).

Segment bütünlüğü ölçümü, sınıf ağırlıklı kayıp, hiperparametre taraması, bileşen çıkarma karşılaştırması ve çoklu seed deneyi bulunmaz. Uzunluk gruplarındaki farklardan uzunluğun nedensel etkisi çıkarılamaz.

## Doğru ve eksik olanı ayırmak

İncelemede temel model akışında doğrulanmış bir dolgu, etiket hizalama veya metrik hesaplama hatası bulunmadı. İki proteinlik kontrolde gerçek uzunluklar, dolgu maskesi, kayıp ve tekli/dolgulu batch çıktıları sınandı. Ardından dört değerlendirme kümesinde tam tahmin çalıştırıldı. Bu kontroller tüm olası girdilerin hatasız işlendiği anlamına gelmez.

Somut sorun, eski sonuç arşivi ile kontrol noktasının farklı epok bilgileri taşımasıydı. Yayında daha yüksek eski skorlar korunup modelin sonucu gibi sunulmadı; aynı ağırlıklardan yeni ölçüler üretildi. Notebook'larda bulunmayan eğitim geçmişi dosyalarına koşulsuz bağımlılıklar da düzeltildi. Yeniden değerlendirme sürümü, uyumlu bir eğitim eğrisi varmış gibi davranmaz.

Özgün rapordaki yorumlar da daraltıldı: testler tam bağımsız değildir; macro F1 artışı bütün sınıfların öğrenildiğini kanıtlamaz; sabit referansın aşılması çift yönlülüğün tek başına katkısını göstermez. Bunlar projenin öğrenme çıktılarıdır: doğru değerlendirme, daha yüksek bir skordan önce gelmelidir.

## Bundan sonra anlamlı olacak deneyler

Bunlar henüz yapılmış çalışmalar değildir: örtüşmesiz/homoloji denetimli bölmelerle duyarlılık ölçümü; aminoasit-koşullu veya tek yönlü referansla kontrollü karşılaştırma; sınıf ağırlıklı kayıp, daha uzun eğitim ve birden fazla seed ile belirsizlik değerlendirmesi. Her yeni koşuda model, veri, ayar ve sonuçların ortak bir deney kimliğiyle ilişkilendirilmesi de eğitim kaydını güçlendirecektir.
