# Veri Kaynakları ve Değerlendirme Sınırları

Bu proje, aminoasit dizisinin her konumu için bir SST8 ikincil yapı etiketi tahmin eder. Veri dosyaları, ödev için sağlanan eğitim, doğrulama ve üç test bölmesi olarak kullanılmıştır. Çalışmanın veri açısından amacı, bu bölmelerin yapısını kontrol etmek, sınıf dağılımını görünür kılmak ve model sonuçlarının hangi koşullarda yorumlanabileceğini açıklamaktır.

## Kaynak ve dosyaların durumu

Kaggle’daki **1. sürüm**, veri kümesini **Open Data Commons Public Domain Dedication and Licence (PDDL)** ile listeliyor. Bu sürümü indirip özgün projedeki beş CSV ile karşılaştırdım. Beş dosya da adları ve içerikleriyle birebir eşleşti. [Kaggle veri kümesi](https://www.kaggle.com/datasets/tamzidhasan/protein-secondary-structure-casp12-cb513-ts115) · [PDDL metni](https://opendatacommons.org/licenses/pddl/1-0/)

### Veriyi hazırlama

1. [Doğrulanan Kaggle arşivini](https://api.kaggle.com/v1/datasets/download/tamzidhasan/protein-secondary-structure-casp12-cb513-ts115?datasetVersionNumber=1) indir ve aç.
2. Beş CSV’yi adlarını koruyarak `run_all.py` dosyasının yanına koy. Dosya adları ve satır sayıları aşağıdaki tabloda.
3. Kurulum ve çalıştırma için [README](../README.md) adımlarını izle. Özgün değerlendirmeden saklanan toplu ölçümler depoda, yeni model ağırlığı ise yerel eğitimle üretilir.

## Şema ve büyüklük

Kullanılan özgün CSV dosyalarında `seq`, `sst3` ve `sst8` sütunları bulunur. Model girdisi `seq`, hedefi `sst8` sütunudur. `sst3` mevcut olmakla birlikte model girdisine veya eğitim hedefine dahil edilmez. Her satır bir protein dizisini, her etiket karakteri ilgili aminoasit konumunu temsil eder.

| Bölme | Dosya | Dizi sayısı | Aminoasit konumu |
| --- | --- | ---: | ---: |
| Eğitim | `training_secondary_structure_train.csv` | 8.678 | 2.221.507 |
| Doğrulama | `validation_secondary_structure_valid.csv` | 2.170 | 556.931 |
| CB513 | `test_secondary_structure_cb513.csv` | 513 | 144.360 |
| TS115 | `test_secondary_structure_ts115.csv` | 115 | 29.704 |
| CASP12 | `test_secondary_structure_casp12.csv` | 21 | 7.256 |

Yerel dosyaların hafif kontrolünde boş dizi, boş SST8 hedefi veya dizi–etiket uzunluğu uyuşmazlığı bulunmamıştır. Bütün bölmelerde SST8 etiket kümesi `B, C, E, G, H, I, S, T` olarak görülür. Eğitim girdisinde `X` dahil 21 farklı karakter vardır; `<PAD>` ve `<UNK>` ile sözlük büyüklüğü 23 olur. Sözlük ve etiket kodlayıcı yalnızca eğitim bölmesinden oluşturulur.

Sınıflar dengeli değildir. Eğitim konumlarının yaklaşık %31,851'i `H`, %24,780'i `C`, %20,447'si `E` iken `B` %0,955 ve `I` %0,503 oranındadır. Bu nedenle yalnızca toplam doğruluk göstermek, nadir sınıflardaki zayıflığı saklayabilir. Macro F1 ve sınıf bazlı ölçümler, sonuç değerlendirmesinin gerekli parçalarıdır.

## Tekrarlar ve bölmeler arası örtüşmeler

CB513 dosyasında 79 fazla tekrar satırı vardır; 513 satır, 434 benzersiz diziyi içerir. Aynı diziye ait farklı SST8 anotasyonu bu bölmenin kendi içinde bulunmamıştır. Tekrarlar, verilen dosyalar korunarak değerlendirme yapıldığı için tutulmuştur. Bu tercih, tekrar eden proteinlerin toplam ölçüme daha fazla ağırlık vermesi sorununu ortadan kaldırmaz.

| Bölme çifti | Tam ortak dizi | Ortak dizilerdeki konum sayısı |
| --- | ---: | ---: |
| Eğitim – CB513 | 2 | 91 |
| Eğitim – TS115 | 1 | 271 |
| TS115 – CASP12 | 4 | 982 |

Diğer bölme çiftlerinde, eğitim–doğrulama dahil, tam dizi eşleşmesi bulunmamıştır. Eğitim–CB513 eşleşmelerinden birinde ve eğitim–TS115 eşleşmesinde SST8 anotasyonları farklıdır; bu gözlem tek başına dosya bozulması olduğunu göstermez. TS115–CASP12 ortak dizilerinin etiketleri aynıdır.

Eğitimle ortak konumlar CB513'ün %0,063'ünü, TS115'in %0,912'sini oluşturur; bu küçük örtüşmelerin bütün performans farkını açıkladığı söylenemez. Öte yandan TS115 ile ortak konumlar, 21 dizilik CASP12'nin %13,534'üdür. Üç test sonucunu tamamen bağımsız tekrarlar gibi sunmak doğru olmaz.

Kontrol yalnızca birebir dizi eşitliğini ölçer. Benzer diziler, homoloji veya protein ailesine göre ayrım yapılmamıştır. Sonuçlar bu sınırlar altında, verilen dosyalar üzerindeki gözlemler olarak okunmalıdır. Örtüşmelerin ayıklandığı ve homolojiye göre ayrılmış yeni bir değerlendirme bu sürümde yapılmış gibi gösterilmez. Model, ölçüm ve geliştirme tartışması [proje raporunda](PROJECT_REPORT.md) yer alır.
