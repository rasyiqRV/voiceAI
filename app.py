import os
import uuid
import io
import shutil
import tempfile
from flask import Flask, request, jsonify, render_template, send_file
try:
    from docx import Document
except ImportError:
    pass # Will be installed
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

# ── Auto-copy logo to static folder
_LOGO_SRC = os.path.join(
    os.path.expanduser("~"),
    ".gemini", "antigravity", "brain",
    "0f1be3c7-678b-42ee-ac7e-64833766186c",
    "media__1775641494024.png"
)
_LOGO_DST = os.path.join(os.path.dirname(__file__), "static", "logo_rkb.png")

if os.path.exists(_LOGO_SRC) and not os.path.exists(_LOGO_DST):
    try:
        shutil.copy2(_LOGO_SRC, _LOGO_DST)
        print(f"[INFO] Logo disalin ke {_LOGO_DST}")
    except Exception as _e:
        print(f"[WARN] Gagal menyalin logo: {_e}")

app = Flask(__name__)

# Max file size: 25MB (Whisper API limit)
app.config['MAX_CONTENT_LENGTH'] = 25 * 1024 * 1024

ALLOWED_EXTENSIONS = {'mp3', 'mp4', 'mpeg', 'mpga', 'm4a', 'wav', 'webm', 'ogg', 'flac'}

# In-memory chunk storage for Vercel (stateless serverless functions
# can't share filesystem between requests)
chunk_storage = {}

client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def allowed_file(filename: str) -> bool:
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def classify_error(error_msg: str) -> str:
    """Classify Groq API errors into user-friendly messages."""
    msg_lower = error_msg.lower()
    if 'api_key' in msg_lower or 'authentication' in msg_lower or 'invalid api key' in msg_lower:
        return 'API Key tidak valid. Periksa GROQ_API_KEY di file .env Anda.'
    elif 'rate limit' in msg_lower:
        return 'Batas penggunaan API tercapai. Coba lagi beberapa saat.'
    elif 'invalid file format' in msg_lower or 'audio' in msg_lower:
        return 'File audio rusak atau format tidak valid. Pastikan file bisa diputar.'
    elif 'timeout' in msg_lower or 'timed out' in msg_lower:
        return 'Proses terlalu lama. Coba file audio yang lebih pendek.'
    else:
        return f'Terjadi kesalahan: {error_msg}'


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/transcribe', methods=['POST'])
def transcribe():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'Tidak ada file yang diunggah.'}), 400

        file = request.files['file']

        chunk_index = int(request.form.get('chunkIndex', 0))
        total_chunks = int(request.form.get('totalChunks', 1))
        file_id = request.form.get('fileId', uuid.uuid4().hex)
        original_filename = request.form.get('filename', file.filename)

        if original_filename == '':
            return jsonify({'error': 'Nama file tidak boleh kosong.'}), 400

        if not allowed_file(original_filename):
            ext = original_filename.rsplit('.', 1)[-1].upper() if '.' in original_filename else 'UNKNOWN'
            return jsonify({'error': f'Format file .{ext} tidak didukung.'}), 415

        # Read chunk data into memory (works on Vercel stateless functions)
        chunk_data = file.read()

        if total_chunks == 1:
            # Single chunk — process directly from memory (most common case)
            audio_bytes = chunk_data
        else:
            # Multi-chunk — store in memory dict
            if file_id not in chunk_storage:
                chunk_storage[file_id] = {}
            chunk_storage[file_id][chunk_index] = chunk_data

            # Not all chunks received yet
            if chunk_index < total_chunks - 1:
                return jsonify({'message': f'Chunk {chunk_index} received.'})

            # All chunks received — merge in memory
            audio_bytes = b''
            for i in range(total_chunks):
                if i not in chunk_storage.get(file_id, {}):
                    # Chunk missing — likely hit different Vercel instance
                    # Clean up and return error
                    chunk_storage.pop(file_id, None)
                    return jsonify({
                        'error': 'Upload gagal: beberapa bagian file hilang. '
                                 'Coba upload ulang atau gunakan file yang lebih kecil (maks 4MB untuk deployment online).'
                    }), 400
                audio_bytes += chunk_storage[file_id][i]
            # Clean up stored chunks
            chunk_storage.pop(file_id, None)

        # Write to temp file for Groq API (it needs a file-like object with name)
        ext = original_filename.rsplit('.', 1)[1].lower() if '.' in original_filename else 'mp3'
        tmp_path = None
        try:
            tmp_fd, tmp_path = tempfile.mkstemp(suffix=f'.{ext}')
            with os.fdopen(tmp_fd, 'wb') as tmp_file:
                tmp_file.write(audio_bytes)

            with open(tmp_path, 'rb') as audio_file:
                transcript = client.audio.transcriptions.create(
                    model="whisper-large-v3",
                    file=audio_file,
                    language="id",
                    response_format="text"
                )

            return jsonify({'transcription': transcript.strip()})

        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except:
                    pass

    except Exception as e:
        return jsonify({'error': classify_error(str(e))}), 500


@app.route('/paraphrase', methods=['POST'])
def paraphrase():
    try:
        data = request.json
        if not data or 'text' not in data:
            return jsonify({'error': 'Teks tidak ditemukan.'}), 400

        text = data['text']

        if not text.strip():
            return jsonify({'error': 'Teks kosong.'}), 400

        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Anda adalah seorang Jurnalis Radio senior yang ahli dalam menulis naskah berita udara (on-air script). "
                        "Tugas Anda adalah mengubah transkrip mentah menjadi BEBERAPA naskah berita radio terpisah yang siap siar "
                        "dengan mengikuti aturan penulisan berikut secara KETAT.\n\n"

                        "## ATURAN UTAMA\n"
                        "1. WAJIB buat MINIMAL 2 berita, idealnya 4 berita dari satu transkrip. "
                        "Pecah berdasarkan topik, sudut pandang, atau aspek berbeda dari isu yang sama.\n"
                        "2. Jika transkrip hanya membahas 1 topik, tetap pecah menjadi minimal 2 berita "
                        "dengan ANGLE (sudut pandang) BERBEDA. Contoh:\n"
                        "   - Berita 1: Fokus pada peristiwa/fakta utama\n"
                        "   - Berita 2: Fokus pada dampak ke masyarakat\n"
                        "   - Berita 3: Fokus pada respons/tanggapan pihak terkait\n"
                        "   - Berita 4: Fokus pada latar belakang/konteks lebih luas\n\n"

                        "## FORMAT JEDA (WAJIB DIPATUHI)\n"
                        "- Gunakan garis miring satu (/) untuk JEDA PENDEK (pengganti koma saat penyiar butuh napas).\n"
                        "- Gunakan garis miring dua (//) untuk AKHIR KALIMAT (pengganti titik).\n"
                        "- Gunakan garis miring tiga (///) sebagai PENUTUP seluruh naskah berita.\n"
                        "- JANGAN PERNAH gunakan tanda titik (.) untuk mengakhiri kalimat / selalu gunakan // sebagai gantinya //\n"
                        "- JANGAN PERNAH gunakan tanda koma (,) jika jeda tersebut untuk napas penyiar / gunakan / sebagai gantinya //\n\n"

                        "## FORMAT JUDUL\n"
                        "Judul menggunakan format: [Nomor Urut]// [Judul Berita]\n"
                        "Judul harus memancing rasa penasaran seperti headline media online:\n"
                        "- Gunakan teknik clickbait yang etis: pertanyaan retoris / fakta mengejutkan / angka spesifik\n"
                        "- Contoh BAGUS: '1// Warga Kaget / Jalan Utama Kota Mendadak Ditutup Tanpa Pemberitahuan'\n"
                        "- Contoh BAGUS: '2// Terungkap / Alasan di Balik Kenaikan Harga Sembako 40 Persen'\n"
                        "- Contoh BURUK: '1// Berita Tentang Jalan Ditutup'\n\n"

                        "## PENYEBUTAN JABATAN\n"
                        "Sebutkan jabatan tokoh SEBELUM namanya.\n"
                        "Contoh: Plt. Bupati Pekalongan / Sukirman //\n"
                        "Contoh: Kepala Dinas Kesehatan / dr. Ahmad //\n\n"

                        "## STRUKTUR NASKAH (4 PARAGRAF)\n"
                        "Gunakan alur berikut untuk SETIAP berita:\n\n"
                        "```\n"
                        "[Nomor Urut]// [Judul Berita]\n\n"
                        "[Paragraf 1 — Ringkasan kejadian: Apa / Siapa / Di mana. "
                        "Lead yang ringkas dan langsung ke inti berita //]\n\n"
                        "[Paragraf 2 — Detail waktu / teknis / atau latar belakang kebijakan. "
                        "Berikan konteks yang membuat pendengar memahami mengapa berita ini penting //]\n\n"
                        "[Paragraf 3 — Kutipan tidak langsung dari narasumber. "
                        "Tulis apa yang disampaikan narasumber tanpa tanda kutip langsung //]\n\n"
                        "Insert - [Nama Tokoh] - [Judul Berita]\n\n"
                        "[Paragraf 4 — Harapan atau dampak ke depan. "
                        "Tutup dengan informasi tentang langkah selanjutnya atau dampak bagi masyarakat //]\n\n"
                        "(---)\n"
                        "///\n"
                        "```\n\n"

                        "## GAYA BAHASA\n"
                        "- Gunakan bahasa TUTUR yang ringkas / jelas / dan mengalir //\n"
                        "- Hindari kalimat yang terlalu panjang tanpa jeda //\n"
                        "- Tulis seperti penyiar sedang BERBICARA kepada pendengar / bukan membaca laporan //\n"
                        "- Gunakan kalimat aktif / bukan pasif //\n"
                        "- Tetap FAKTUAL — jangan menambah informasi yang tidak ada di transkrip //\n\n"

                        "## CONTOH NASKAH LENGKAP\n"
                        "```\n"
                        "1// Warga Terkejut / Jalan Protokol Kota Pekalongan Mendadak Ditutup\n\n"
                        "Warga Kota Pekalongan / dikejutkan dengan penutupan mendadak Jalan Protokol utama / "
                        "tepat di jantung kota // Penutupan ini dilakukan tanpa pemberitahuan resmi kepada masyarakat //\n\n"
                        "Penutupan jalan tersebut berlangsung sejak Senin pagi / dan diperkirakan akan berlangsung "
                        "selama dua pekan ke depan // Hal ini terkait dengan proyek revitalisasi trotoar "
                        "yang digagas oleh Pemerintah Kota //\n\n"
                        "Kepala Dinas Pekerjaan Umum / Bambang Sutrisno / menyatakan bahwa penutupan ini "
                        "memang harus dilakukan demi kelancaran proyek // Pihaknya mengaku telah mengirimkan "
                        "surat pemberitahuan ke kelurahan setempat / namun informasi tersebut belum sampai "
                        "ke seluruh warga //\n\n"
                        "Insert - Bambang Sutrisno - Jalan Protokol Ditutup\n\n"
                        "Masyarakat berharap agar ke depannya / pemerintah dapat memberikan sosialisasi "
                        "yang lebih luas sebelum melakukan penutupan jalan // Sehingga warga bisa "
                        "mempersiapkan jalur alternatif //\n\n"
                        "(---)\n"
                        "///\n"
                        "```\n\n"

                        "## PENTING\n"
                        "- JANGAN gabung semua topik jadi 1 berita / WAJIB pisahkan //\n"
                        "- Setiap berita harus bisa BERDIRI SENDIRI //\n"
                        "- Setiap berita WAJIB diakhiri dengan inisial (---) lalu /// //\n"
                        "- Insert audio WAJIB ada di setiap berita / ditempatkan setelah paragraf 3 (kutipan tidak langsung) "
                        "dan sebelum paragraf 4 (penutup) //\n"
                        "- Jangan gunakan markdown formatting (** / ## / dll) / tulis plain text saja //\n"
                        "- INGAT: TIDAK ADA titik (.) dan koma (,) di seluruh naskah / hanya gunakan / dan // //"
                    )
                },
                {
                    "role": "user",
                    "content": text
                }
            ],
            temperature=0.75,
            max_tokens=4096,
        )
        paraphrased_text = completion.choices[0].message.content
        return jsonify({'paraphrased': paraphrased_text.strip()})

    except Exception as e:
        error_msg = str(e)
        if 'api_key' in error_msg.lower() or 'authentication' in error_msg.lower() or 'invalid api key' in error_msg.lower():
            message = 'API Key tidak valid. Periksa GROQ_API_KEY di file .env Anda.'
        elif 'rate limit' in error_msg.lower():
            message = 'Batas penggunaan API tercapai. Coba lagi beberapa saat.'
        else:
            message = f'Terjadi kesalahan saat menyusun naskah: {error_msg}'
        return jsonify({'error': message}), 500


@app.route('/export-docx', methods=['POST'])
def export_docx():
    try:
        data = request.json
        if not data or 'text' not in data:
            return jsonify({'error': 'Teks tidak ditemukan.'}), 400

        text = data['text']
        filename = data.get('filename', 'VoiceScript_Document')

        document = Document()
        for paragraph in text.split('\n'):
            if paragraph.strip():
                document.add_paragraph(paragraph.strip())

        f = io.BytesIO()
        document.save(f)
        f.seek(0)

        return send_file(
            f,
            as_attachment=True,
            download_name=f"{filename}.docx",
            mimetype='application/vnd.openxmlformats-officedocument.wordprocessingml.document'
        )
    except Exception as e:
        return jsonify({'error': f'Gagal membuat dokumen: {str(e)}'}), 500


@app.errorhandler(413)
def file_too_large(e):
    return jsonify({'error': 'File terlalu besar. Batas maksimal adalah 25MB.'}), 413


if __name__ == '__main__':
    app.run(debug=True, port=5000)
