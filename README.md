# VoiceScript
### Transkripsi Audio ke Teks Bahasa Indonesia

Website berbasis Python Flask yang menggunakan **Groq Whisper** (gratis & cepat) untuk mentranskripsi file audio ke teks Bahasa Indonesia dengan akurasi tinggi.

## Cara Menggunakan

1. **Upload file audio** — seret & lepas atau klik area upload
2. **Preview audio** — dengarkan file sebelum diproses
3. **Klik "Transkripsi Sekarang"** — AI memproses audio via Groq
4. **Dapatkan hasil** — salin atau unduh sebagai file `.txt`

---

## Format Audio yang Didukung

| Format | Keterangan |
|--------|-----------|
| MP3    | Paling umum |
| WAV    | Kualitas tinggi |
| M4A    | Apple/iPhone |
| OGG    | Open source |
| FLAC   | Lossless |
| WEBM   | Web recording |

> Batas ukuran file: **25 MB** (batas Whisper API)

---

## Privasi

File audio yang diunggah hanya disimpan sementara di folder `tmp_uploads/` selama proses transkripsi berlangsung dan langsung dihapus setelah selesai.
