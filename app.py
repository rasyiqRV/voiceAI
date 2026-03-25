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

client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def allowed_file(filename: str) -> bool:
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/transcribe', methods=['POST'])
def transcribe():
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
        return jsonify({'error': f'Format file .{ext} tidak didukung. Gunakan MP3, WAV, M4A, OGG, FLAC, atau WEBM.'}), 415

    # Save to temp file
    tmp_dir = os.path.join(tempfile.gettempdir(), 'VoiceScriptUploads', file_id)
    os.makedirs(tmp_dir, exist_ok=True)
    
    chunk_path = os.path.join(tmp_dir, f"part_{chunk_index}")
    file.save(chunk_path)
    
    # If not all chunks received yet
    if chunk_index < total_chunks - 1:
        return jsonify({'message': f'Chunk {chunk_index} received.'})

    # All chunks received
    ext = original_filename.rsplit('.', 1)[1].lower() if '.' in original_filename else 'mp3'
    final_path = os.path.join(tmp_dir, f"final_{file_id}.{ext}")

    try:
        # Merge all chunks
        with open(final_path, 'wb') as outfile:
            for i in range(total_chunks):
                part_path = os.path.join(tmp_dir, f"part_{i}")
                with open(part_path, 'rb') as infile:
                    outfile.write(infile.read())
                os.remove(part_path)

        with open(final_path, 'rb') as audio_file:
            transcript = client.audio.transcriptions.create(
                model="whisper-large-v3",   # Groq: free, faster, higher quality
                file=audio_file,
                language="id",              # Force Indonesian
                response_format="text"
            )

        return jsonify({'transcription': transcript.strip()})

    except Exception as e:
        error_msg = str(e)
        if 'api_key' in error_msg.lower() or 'authentication' in error_msg.lower() or 'invalid api key' in error_msg.lower():
            message = 'API Key tidak valid. Periksa GROQ_API_KEY di file .env Anda.'
        elif 'rate limit' in error_msg.lower():
            message = 'Batas penggunaan API tercapai. Coba lagi beberapa saat.'
        elif 'invalid file format' in error_msg.lower() or 'audio' in error_msg.lower():
            message = 'File audio rusak atau format tidak valid. Pastikan file bisa diputar.'
        else:
            message = f'Terjadi kesalahan: {error_msg}'
        return jsonify({'error': message}), 500

    finally:
        if os.path.exists(final_path):
            try:
                os.remove(final_path)
            except:
                pass
        # Clean up the temp directory for this file
        try:
            os.rmdir(tmp_dir)
        except:
            pass


@app.route('/paraphrase', methods=['POST'])
def paraphrase():
    data = request.json
    if not data or 'text' not in data:
        return jsonify({'error': 'Teks tidak ditemukan.'}), 400

    text = data['text']
    
    if not text.strip():
        return jsonify({'error': 'Teks kosong.'}), 400

    try:
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Ubah teks transkrip mentah dari user menjadi naskah berita terpisah berdasarkan topik.\n\n"
                        "Ikuti format penulisan naskah berita radio/TV yang ketat ini:\n"
                        "1. Jika ada beberapa topik, pisahkan menjadi beberapa berita (berikan nomor seperti '1 // Judul Berita', '2 // Judul Berita').\n"
                        "2. Gunakan tanda '/' sebagai pengganti koma (jeda singkat) dan '//' sebagai pengganti titik (jeda panjang).\n"
                        "3. Paragraf pertama HARUS deduktif (menyampaikan inti berita keseluruhan).\n"
                        "4. Paragraf berikutnya menjelaskan rincian dari apa yang disampaikan di paragraf pertama secara lugas, ringkas, dan to the point.\n"
                        "5. Setelah rincian, buat baris baru untuk kutipan narasumber dengan format: 'Insert -- [Nama Narasumber] - [Judul Berita]'.\n"
                        "6. Lanjutkan dengan paragraf pernyataan dari narasumber.\n"
                        "7. dibuat agar punya story telling yang bagus.\n"
                        "8. Akhiri setiap berita dengan tanda '/// (Nama Reporter)', gunakan '(----)' sebagai default reporter."
                    )
                },
                {
                    "role": "user",
                    "content": text
                }
            ],
            temperature=0.7,
            max_tokens=1024,
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


@app.errorhandler(413)
def file_too_large(e):
    return jsonify({'error': 'File terlalu besar. Batas maksimal adalah 25MB.'}), 413


if __name__ == '__main__':
    app.run(debug=True, port=5000)
