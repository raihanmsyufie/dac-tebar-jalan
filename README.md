# Balik Modal Pendidikan, Script Analisis DAC Tebar Jalan

Script analisis di balik dashboard tim Tebar Jalan untuk DAC Data Slayer 4.0. Pertanyaannya, lulusan SMP atau SMA lebih untung bekerja atau lanjut sekolah? Tiap jenjang pendidikan dinilai seperti investasi, dengan untung bersih (NPV), imbal per tahun (IRR), dan lama modal kembali (payback) per provinsi.

Halaman notebook: https://raihanmsyufie.github.io/dac-tebar-jalan/

Glosarium istilah dashboard ada di bagian 4 notebook, atau langsung lewat https://raihanmsyufie.github.io/dac-tebar-jalan/#glosarium

## Isi repo

| File | Isi |
|---|---|
| `index.html` | Halaman pembungkus, panel samping plus notebook selebar layar |
| `analisis_final_ide_A.ipynb` | Notebook Python sumber |
| `analisis_final_ide_A.py` | Notebook yang sama dalam bentuk script |
| `analisis_final_ide_A.html` | Hasil notebook yang ditampilkan halaman |
| `*.csv` | Data BPS yang dibaca notebook |

## Sumber data

Semua data dari Badan Pusat Statistik (bps.go.id).

| Data | Sumber |
|---|---|
| Upah buruh menurut provinsi dan pendidikan | Publikasi Keadaan Pekerja di Indonesia, Agustus 2024 dan 2025 |
| Angkatan kerja dan penganggur menurut provinsi dan pendidikan | Publikasi Keadaan Angkatan Kerja di Indonesia, Agustus 2024 dan 2025 |
| Biaya pendidikan, komponennya, dan penerima beasiswa | Statistik Penunjang Pendidikan 2024 (Susenas MSBP 2024) |
| Upah, TPT, setengah pengangguran, pekerja informal, dan upah per jam menurut pendidikan | Tabel dinamis Sakernas 2015 sampai 2025 |
| Upah menurut kelompok umur dan IHK subkelompok pendidikan | Tabel dinamis BPS |

## Cara menjalankan ulang

1. Unduh seluruh isi repo ke satu folder.
2. Buka `analisis_final_ide_A.ipynb` di VS Code atau Jupyter, lalu jalankan semua sel. Paket yang belum ada (pandas, numpy, matplotlib, scipy) dipasang otomatis oleh sel pertama.
