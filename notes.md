# Source

https://www.kaggle.com/code/infocusp/diffstride-explainable-notebook

# Overview

Mendeskripsikan intisari suatu ide dari artikel ilmiah ["Learning strides in convolutional neural network"](doc:https://arxiv.org/pdf/2202.01653.pdf) bersamaan dengan beberapa ide latar belakang dan visualisasinya.
artikel ilmiah tersebut adalah salah satu dari banyak artikel ilmiah yang menerima penghargaan ["outstanding paper awards"](doc:https://blog.iclr.cc/2022/04/20/announcing-the-iclr-2022-outstanding-paper-award-recipients/) pada event ICLR 2022

Dengan asumsi pembaca sudah mengenal dasar dari arsitektur CNN. Jika pembaca masih belum mengetahui mengenai dasar dari arsitektur CNN, maka dapat mempelajari melalui [materi berikut](doc:https://cs231n.github.io/) pada event ICLR 2022

# Penjelasan singkat mengenai artikel ilmiah

- Arsitektur pada Model CNN umumnya memiliki beberapa tahapan downsampling yang secara progresif mengurangi resolusi dari representasi perantara dimana prosesnya akan dilanjutkan lebih dalam ke jaringan (network)
- Tujuan dari downsampling adalah memberikan beberapa pergeseran invarian dan mengurangi jumlah komputasi yang diperlukan oleh arsitektur dan disisi lain juga mempertahankan informasi penting dari dataset original citra
- Salah satu hyperparameter dari lapisan Convolution pada CNN yang mengontrol jumlah downsampling adalah stride. Stride pada dasarnya adalah suatu langkah ukuran (dalam arah horizontal dan vertikal) yang diterapkan di lapisan Convolution saat memutar (convolving) suatu citra
- Sebagai contoh: stride = 1 memindahkan filter sejumlah 1 langkah dalam satu waktu saat pemutaran, stride = 2 memindahkan filter sejumlah 2 langkah dalam satu waktu saat pemutaran
- Dikarenakan ukuran stride tidak dapat dibedakan (not differentiable), mencari konfigurasi terbaik dari suatu stride untuk layer yang berbeda membutuhkan proses _crossvalidation_ atau _discrete optimization_. Sebagai contoh pada studi kasus untuk menemukan aristektur terbaik, dengan cara mencoba berbagai macam nilai dan memilih salah satu dari berbagai nilai tersebut yang memberikan hasil yang terbaik, dimana akan menghabiskan lebih banyak waktu dalam proses training.

Interpretasi visual dari strides

![Strides 0](/asset/strides_0.jpg)

![Strides 1](/asset/strides_1.jpg)

![Strides 2](/asset/strides_2.jpg)

[Ref: Apa yang dimaksud dengan "stride" pada Convolutional Neural Network](doc:https://medium.com/machine-learning-algorithms/what-is-stride-in-convolutional-neural-network-e3b4ae9baedb)

Langkah serupa pada perhitungan strides juga berlaku saat melakukan operasi pooling (pooling operation)

Apakah ada cara lain untuk menentukan suatu stride yang optimal menggunakan algoritma
optimisasi seperti _gradient descent_ (sehingga memungkinkan untuk melakukan optimisasi dengan biaya komputasi yang lebih rendah)

- Jawaban dari pertanyaan tersebut adalah menggunakan Metode Diffstride. Merupakan metode yang mengenalkan teknik downsampling dengan stride yang langkah yang dapat dipelajari (learnable strides). Sebuah lapisan yang dapat dipelajari (learnable layer) yang menentukan ukuran dari _cropping mask_ (yang diimplementasikan pada representasi domain _fourier_), sehingga dapat mengubah ukuran dengan cara yang dapat dibedakan (differentiable). Cara kerja dari diffstride secara detail akan dijabarkan lebih lanjut. Namun terlebih dahulu ditampilkan hasil atau outcome dalam penggunaan diffstride terlebih dahulu.
- Artikel ilmiah ini mendemonstrasikan efektivitas dari penggunaan DiffStride (beserta kemampuannya secara general) terhadap kasus klasifikasi citra. Hasil yang dijabarkan dari artikel ilmiah tersebut ditampilkan pada tabel dibawah. Berdasarkan pada tabel dibawah menunjukkan bahwa penerapan DiffStride pada Model Arsitektur ResNet18 mengungguli semua Teknik Downsampling dalam penerapannya pada Model Arsitektur ResNet18 standar.

| Init. Strides | Strided Conv. | Spectral   | DiffStride | Strided Conv. | Spectral   | DiffStride |
| ------------- | ------------- | ---------- | ---------- | ------------- | ---------- | ---------- |
| (2, 2, 2)     | 91.4 ± 0.2    | 92.4 ± 0.1 | 92.5 ± 0.1 | 66.8 ± 0.2    | 73.7 ± 0.1 | 73.4 ± 0.5 |
| (2, 2, 3)     | 90.5 ± 0.1    | 92.2 ± 0.2 | 92.8 ± 0.1 | 63.4 ± 0.5    | 73.7 ± 0.2 | 73.5 ± 0.0 |
| (1, 3, 1)     | 90.0 ± 0.4    | 91.1 ± 0.1 | 92.4 ± 0.1 | 64.9 ± 0.5    | 70.3 ± 0.3 | 73.4 ± 0.2 |
| (3, 1, 3)     | 85.7 ± 0.1    | 90.9 ± 0.2 | 92.4 ± 0.1 | 55.3 ± 0.8    | 69.4 ± 0.4 | 73.7 ± 0.4 |
| (3, 1, 2)     | 86.4 ± 0.1    | 90.9 ± 0.2 | 92.3 ± 0.1 | 56.2 ± 0.3    | 69.9 ± 0.2 | 73.4 ± 0.3 |
| (3, 2, 3)     | 82.0 ± 0.6    | 89.2 ± 0.2 | 92.3 ± 0.1 | 48.2 ± 0.2    | 66.6 ± 0.5 | 73.6 ± 0.4 |
| **Mean Acc:** | 87.7 ± 3.4    | 91.1 ± 1.1 | 92.4 ± 0.2 | 59.1 ± 6.7    | 70.6 ± 2.6 | 73.5 ± 0.3 |

# Beberapa Persyaratan

Untuk memahami intisari suatu ide dibalik diffstride, penting bagi pembaca untuk memahami konsep berikut:

- FFT - _Fourier Domain Representation of Image_
- Dualitas antara _convolution_ dalam domain citra dan perkalian elemen dalam _domain fouriers_

Jika sudah familiar terhadap konsep tersebut, maka bagian ini dapat di skip.

# Fourier Transform

Teorema Fourier menyatakan bahwa: setiap fungsi kontinu (continous function) dapat direpresentasikan sebagai penjumlahan berbobot tak terbatas (infinte weighted summation) dari gelombang sinus dan cosinus dengan frekuensi yang berbeda. Bobot (weight coefficients) dari gelombang sinus dan cosinus ini merupakan representasi domain frekuensi dari sinyal yang diberikan.

Cermati kasus sinyal 1D dibawah ini.

Sebagai contoh, diketahui gelombang sinus dari 2 frekuensi konstituen, membentuk sinyal yang diberikan. Kemudian pada domain waktu, sinyal campuran (mixed signal) akan direpresentasikan sebagai penambahan amplitudo dua sinyal sinusodial.

Kemudian kita mengimplementasikan DFT pada sinyal campuran (mixed signal). Sehingga sebagai representasi _fourier domain_, kita akan memperoleh puncak dari frekuensi konstituen yang merepresentasikan amplitudonya dalam domain waktu.

![Discrete Fourier Transform](/asset/fourier_transform.jpg)

[Ref: Issac's science blog](doc:https://isaacscienceblog.com/2017/08/13/fourier-transform/)

# Discrete Fourier Transform (DFT) of images

Sinyal dalam kehidupan nyata bersifat kontinu (continous) dalam periode waktu tertentu. Apa yang kita amati merupakan rangkaian variasi cahaya.

Namun, saat mengkonversinya menjadi bentuk citra, ditangkap versi diskrit yang sama. Sehingga citra apa pun akan direpresentasikan sebagai array MxN dari nilai bilangan bulat.

Mengkonversi sinyal diskrit seperti ini ke domain fourier memerlukan _Discrete Time Fourier Transform(DTFT)_. Namun, output dari DTFT adalah fungsi kontinu periodik dalam domain _fourier_. Sehingga untuk melakukan komputasi/merepresentasikan dalam bentuk digital, kita mengambil sampel dari representasi berkelanjutan ini. Dimana disebut sebagai _Discrete Fourier Transform (DFT)_.

Sinyal _Discrete_ yang bersifat non periodik dapat didekati/diwakili (approximated/represented) dalam domain _fourier_ menggunakan koefisien DFT. Pada DFT apa yang akan kita lakukan sebenarnya adalah mencari _Discrete Time Fourier Transform Approximation_ pada interval dimana fungsi yang digunakan bersifat periodik.

_Fast Fourier Transform_ adalah cara/algoritma paralel yang efisien secara komputasi untuk menghitung DFT. FFT (_Fast Fourier Transform_) merupakan inti dari kompresi citra/audio/sinyal. Merupakan salah satu algoritma yang powerful dimasa saat ini. FFT merupakan cara untuk menghitung DFT.

Fourier Transform digunakan untuk menganalisis karakteristik frekuensi dari berbagai macam filter. Untuk citra, 2D _Discrete Fourier Transform (DFT)_ digunakan untuk mencari domain frekuensi. Detail terkait dengan hal tersebut dapat ditemukan pada buku yang membahas mengenai _image processing_ atau _signal processing_. Mari kita amati dengan menjalankan percobaan berikut

# Visualizing FFT

Pertama kita melakukan perhitungan DFT pada suatu citra menggunakan Algoritma FFT (keluar dari kotak opencv). DFT merupakan komponen nyata dan _imaginary_ (yang direpresentasikan oleh 2 channels). Kita akan mengkomputasikan magnitude menggunakan komponen tersebut dan memplot spektrum yang dihasilkan.

# Spectral Pooling

Inti dari FFT adalah sebagian besar konten gambar terkandung dalam bagian frekuensi rendah. Komponen frekuensi tinggi mewakili tepi dan variasi pada citra. Sehingga saat diterapkan _low pass filtering_ pada suatu citra, kita mempertahankan sebagian besar informasi tetap utuh bahkan saat membuang sebagian besar konten frekuensi tinggi dan mengurangi dimensi dalam jumlah besar. Ini adalah ide dari _spectral pooling_ dan digunakan dalam kompresi menggunakan langkah-langkah berikut.

- Mengkonversi citra ke domain fourier menggunakan FFT
- Mengekstrasi frekuensi rendah (terkompresi) kecil dari representasi fourier
- Melakukan inverse FFT kembali ke domain citra (spasial).

Tidak seperti spatial pooling yang membutuhkan stride dengan tipe data integer, spectral pooling hanya membutuhkan dimensi output integer

Berdasarkan hasil observasi, hal yang menarik dari IDFT (_magnitude spectrum/low-pass-filter_) pada citra, adalah bahwa meskipun menggunakan dimensi kecil dari suatu data (60 out 256), kebanyakan informasi masih dapat diperoleh.
