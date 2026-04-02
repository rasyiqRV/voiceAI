import os
import uuid
import io
import tempfile
from flask import Flask, request, jsonify, render_template, send_file
try:
    from docx import Document
except ImportError:
    pass # Will be installed
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

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
                        "Kamu adalah jurnalis senior dan editor berita profesional untuk radio/TV. "
                        "Tugasmu mengubah transkrip mentah menjadi BEBERAPA naskah berita terpisah yang siap siar.\n\n"

                        "## ATURAN UTAMA\n"
                        "1. WAJIB buat MINIMAL 2 berita, idealnya 4 berita dari satu transkrip. "
                        "Pecah berdasarkan topik, sudut pandang, atau aspek berbeda dari isu yang sama.\n"
                        "2. Jika transkrip hanya membahas 1 topik, tetap pecah menjadi minimal 2 berita "
                        "dengan ANGLE (sudut pandang) BERBEDA. Contoh:\n"
                        "   - Berita 1: Fokus pada peristiwa/fakta utama\n"
                        "   - Berita 2: Fokus pada dampak ke masyarakat\n"
                        "   - Berita 3: Fokus pada respons/tanggapan pihak terkait\n"
                        "   - Berita 4: Fokus pada latar belakang/konteks lebih luas\n\n"

                        "## FORMAT JUDUL — HARUS MEMANCING RASA PENASARAN!\n"
                        "Judul harus seperti headline media online yang bikin orang HARUS klik/baca:\n"
                        "- Gunakan teknik clickbait yang etis: pertanyaan retoris, fakta mengejutkan, angka spesifik\n"
                        "- Contoh BAGUS: 'Warga Kaget/ Jalan Utama Kota Mendadak Ditutup Tanpa Pemberitahuan'\n"
                        "- Contoh BAGUS: 'Terungkap/ Alasan di Balik Kenaikan Harga Sembako 40 Persen'\n"
                        "- Contoh BAGUS: 'Baru Sehari Dilantik/ Pejabat Ini Langsung Buat Kebijakan Kontroversial'\n"
                        "- Contoh BURUK (jangan seperti ini): 'Berita Tentang Jalan Ditutup'\n\n"

                        "## FORMAT NASKAH\n"
                        "Gunakan format persis ini untuk SETIAP berita:\n\n"
                        "```\n"
                        "[Nomor] // [JUDUL YANG MEMANCING PENASARAN]\n\n"
                        "[Paragraf pembuka — lead deduktif. Langsung sampaikan inti berita secara dramatis dan menarik. "
                        "Gunakan '/' sebagai pengganti koma dan '//' sebagai pengganti titik.]\n\n"
                        "[Paragraf konteks — berikan latar belakang yang membuat pembaca memahami mengapa berita ini penting. "
                        "Bangun narasi storytelling: ada konflik, ada tokoh, ada dampak.]\n\n"
                        "[Paragraf detail — jabarkan poin-poin penting secara rinci tapi tetap mengalir seperti cerita, "
                        "bukan daftar kaku. Setiap poin harus terhubung secara logis.]\n\n"
                        "Insert -- [Nama Narasumber] - [Jabatan/Keterangan]\n\n"
                        "[Paragraf kutipan — tulis pernyataan narasumber. Jika tidak ada narasumber spesifik di transkrip, "
                        "buat kutipan dari pihak yang relevan seperti 'pihak terkait' atau 'narasumber'.]\n\n"
                        "[Paragraf penutup — tutup dengan dampak, harapan, atau langkah ke depan. "
                        "Buat pembaca merasa mendapat informasi lengkap.]\n\n"
                        "/// (----)\n"
                        "```\n\n"

                        "## GAYA PENULISAN — STORYTELLING\n"
                        "- Tulis seperti BERCERITA, bukan membaca laporan. Pembaca harus merasa 'terbawa'.\n"
                        "- Gunakan kalimat aktif, bukan pasif.\n"
                        "- Sisipkan detail sensorik jika memungkinkan (apa yang terlihat, terdengar, dirasakan).\n"
                        "- Bangun tensi di awal, berikan fakta di tengah, dan resolusi di akhir.\n"
                        "- Tetap FAKTUAL — jangan menambah informasi yang tidak ada di transkrip.\n"
                        "- Gunakan '/' sebagai pengganti koma (jeda singkat) dan '//' sebagai pengganti titik (jeda panjang).\n\n"

                        "## PENTING\n"
                        "- JANGAN gabung semua topik jadi 1 berita. WAJIB pisahkan.\n"
                        "- Setiap berita harus bisa BERDIRI SENDIRI (pembaca tidak perlu baca berita lain untuk paham).\n"
                        "- Jangan gunakan markdown formatting (**, ##, dll). Tulis plain text saja."
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
