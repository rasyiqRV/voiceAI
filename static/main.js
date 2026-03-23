/* =============================================
   VoiceScript — Main JavaScript Logic
   ============================================= */

const dropZone       = document.getElementById('drop-zone');
const fileInput      = document.getElementById('file-input');
const transcribeBtn  = document.getElementById('transcribe-btn');
const btnText        = document.getElementById('btn-text');
const btnSpinner     = document.getElementById('btn-spinner');
const uploadIcon     = document.getElementById('upload-icon');
const fileIcon       = document.getElementById('file-icon');
const dropLabel      = document.getElementById('drop-zone-label');
const dropHint       = document.getElementById('drop-zone-hint');
const audioPreview   = document.getElementById('audio-preview');
const audioPlayer    = document.getElementById('audio-player');
const audioName      = document.getElementById('audio-name');
const audioSize      = document.getElementById('audio-size');
const removeFileBtn  = document.getElementById('remove-file');
const resultCard     = document.getElementById('result-card');
const resultText     = document.getElementById('result-text');
const wordCount      = document.getElementById('word-count');
const copyBtn        = document.getElementById('copy-btn');
const downloadBtn    = document.getElementById('download-btn');
const errorCard      = document.getElementById('error-card');
const errorMessage   = document.getElementById('error-message');
const retryBtn       = document.getElementById('retry-btn');
const paraphraseBtn  = document.getElementById('paraphrase-btn');
const newsCard       = document.getElementById('news-card');
const newsText       = document.getElementById('news-text');
const newsCopyBtn    = document.getElementById('news-copy-btn');
const newsDownloadBtn= document.getElementById('news-download-btn');

let selectedFile = null;
let lastFilename = '';

// ── File Selection ─────────────────────────────

dropZone.addEventListener('click', (e) => {
    if (e.target.closest('#remove-file')) return;
    fileInput.click();
});

dropZone.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' || e.key === ' ') fileInput.click();
});

fileInput.addEventListener('change', () => {
    if (fileInput.files.length > 0) handleFileSelect(fileInput.files[0]);
});

// ── Drag & Drop ────────────────────────────────

['dragenter', 'dragover'].forEach(event => {
    dropZone.addEventListener(event, (e) => {
        e.preventDefault();
        e.stopPropagation();
        dropZone.classList.add('dragging');
    });
});

['dragleave', 'drop'].forEach(event => {
    dropZone.addEventListener(event, (e) => {
        e.preventDefault();
        e.stopPropagation();
        dropZone.classList.remove('dragging');
    });
});

dropZone.addEventListener('drop', (e) => {
    const files = e.dataTransfer.files;
    if (files.length > 0) handleFileSelect(files[0]);
});

// ── Handle File ────────────────────────────────

function handleFileSelect(file) {
    selectedFile = file;
    lastFilename = file.name;

    // Update drop zone visuals
    uploadIcon.classList.add('hidden');
    fileIcon.classList.remove('hidden');
    dropLabel.textContent = 'File audio siap diunggah';
    dropHint.textContent  = 'Klik untuk mengganti file';

    // Show audio preview
    audioName.textContent = file.name;
    audioSize.textContent = formatFileSize(file.size);
    const objectUrl = URL.createObjectURL(file);
    audioPlayer.src = objectUrl;
    audioPreview.classList.remove('hidden');

    // Enable button
    transcribeBtn.disabled = false;
    btnText.textContent = 'Transkripsi Sekarang';

    // Hide previous results
    hideResult();
    hideError();
}

function resetDropZone() {
    selectedFile = null;
    fileInput.value = '';
    uploadIcon.classList.remove('hidden');
    fileIcon.classList.add('hidden');
    dropLabel.textContent = 'Seret & lepas file audio di sini';
    dropHint.textContent  = 'atau klik untuk memilih file';
    audioPreview.classList.add('hidden');
    audioPlayer.src = '';
    transcribeBtn.disabled = true;
    btnText.textContent = 'Pilih File untuk Memulai';
}

removeFileBtn.addEventListener('click', (e) => {
    e.stopPropagation();
    resetDropZone();
    hideResult();
    hideError();
});

retryBtn.addEventListener('click', () => {
    hideError();
});

// ── Transcription ──────────────────────────────

transcribeBtn.addEventListener('click', async () => {
    if (!selectedFile) return;
    setLoading(true);
    hideResult();
    hideError();

    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
        const response = await fetch('/transcribe', {
            method: 'POST',
            body: formData,
        });

        const data = await response.json();

        if (!response.ok || data.error) {
            throw new Error(data.error || 'Terjadi kesalahan pada server.');
        }

        showResult(data.transcription, lastFilename);

    } catch (err) {
        showError(err.message);
    } finally {
        setLoading(false);
    }
});

// ── Result Display ─────────────────────────────

function showResult(text, filename) {
    resultText.textContent = '';
    resultCard.classList.remove('hidden');
    resultCard.scrollIntoView({ behavior: 'smooth', block: 'start' });

    // Typewriter animation
    let i = 0;
    const speed = Math.max(8, Math.min(25, Math.floor(8000 / text.length)));
    const interval = setInterval(() => {
        resultText.textContent += text[i] || '';
        i++;
        if (i >= text.length) {
            clearInterval(interval);
            updateWordCount(text);
        }
    }, speed);

    // Download setup
    downloadBtn.onclick = () => downloadDocx(text, filename.replace(/\.[^.]+$/, '') + '_transkripsi');
}

function updateWordCount(text) {
    const words = text.trim().split(/\s+/).filter(w => w.length > 0).length;
    const chars = text.length;
    wordCount.textContent = `${words} kata · ${chars} karakter`;
}

function hideResult() {
    resultCard.classList.add('hidden');
    resultText.textContent = '';
    if (typeof hideNews === 'function') hideNews();
}

// ── Paraphrase ─────────────────────────────────

paraphraseBtn.addEventListener('click', async () => {
    const textToParaphrase = resultText.textContent;
    if (!textToParaphrase) return;

    // Loading state
    const origHtml = paraphraseBtn.innerHTML;
    paraphraseBtn.innerHTML = '<div class="spinner" style="display:inline-block;width:16px;height:16px;border-width:2px;border-color:currentColor currentColor transparent transparent;margin-right:8px;"></div> Menyusun...';
    paraphraseBtn.style.display = 'flex';
    paraphraseBtn.style.alignItems = 'center';
    paraphraseBtn.disabled = true;
    
    hideNews();
    hideError();

    try {
        const response = await fetch('/paraphrase', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text: textToParaphrase })
        });

        const data = await response.json();

        if (!response.ok || data.error) {
            throw new Error(data.error || 'Terjadi kesalahan saat menyusun naskah.');
        }

        showNews(data.paraphrased, lastFilename);

    } catch (err) {
        showError(err.message);
    } finally {
        paraphraseBtn.innerHTML = origHtml;
        paraphraseBtn.disabled = false;
        paraphraseBtn.style.display = '';
        paraphraseBtn.style.alignItems = '';
    }
});

function showNews(text, filename) {
    newsText.textContent = '';
    newsCard.classList.remove('hidden');
    newsCard.scrollIntoView({ behavior: 'smooth', block: 'start' });

    // Typewriter animation
    let i = 0;
    const speed = Math.max(8, Math.min(25, Math.floor(8000 / text.length)));
    const interval = setInterval(() => {
        newsText.textContent += text[i] || '';
        i++;
        if (i >= text.length) {
            clearInterval(interval);
        }
    }, speed);

    // Download setup
    newsDownloadBtn.onclick = () => downloadDocx(text, filename.replace(/\.[^.]+$/, '') + '_naskah');
}

function hideNews() {
    newsCard.classList.add('hidden');
    newsText.textContent = '';
}

// ── Error Display ──────────────────────────────

function showError(msg) {
    errorMessage.textContent = msg;
    errorCard.classList.remove('hidden');
    errorCard.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function hideError() {
    errorCard.classList.add('hidden');
}

// ── Copy to Clipboard ──────────────────────────

copyBtn.addEventListener('click', async () => {
    const text = resultText.textContent;
    if (!text) return;
    try {
        await navigator.clipboard.writeText(text);
        const orig = copyBtn.innerHTML;
        copyBtn.innerHTML = `<svg viewBox="0 0 24 24" fill="none" width="16" height="16">
            <path d="M5 13l4 4L19 7" stroke="#34d399" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
        </svg> Tersalin!`;
        copyBtn.style.borderColor = 'rgba(52,211,153,0.4)';
        copyBtn.style.color = '#34d399';
        setTimeout(() => {
            copyBtn.innerHTML = orig;
            copyBtn.style.borderColor = '';
            copyBtn.style.color = '';
        }, 2000);
    } catch {
        showError('Gagal menyalin teks. Coba pilih teks secara manual.');
    }
});

newsCopyBtn.addEventListener('click', async () => {
    const text = newsText.textContent;
    if (!text) return;
    try {
        await navigator.clipboard.writeText(text);
        const orig = newsCopyBtn.innerHTML;
        newsCopyBtn.innerHTML = `<svg viewBox="0 0 24 24" fill="none" width="16" height="16">
            <path d="M5 13l4 4L19 7" stroke="#34d399" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
        </svg> Tersalin!`;
        newsCopyBtn.style.borderColor = 'rgba(52,211,153,0.4)';
        newsCopyBtn.style.color = '#34d399';
        setTimeout(() => {
            newsCopyBtn.innerHTML = orig;
            newsCopyBtn.style.borderColor = '';
            newsCopyBtn.style.color = '';
        }, 2000);
    } catch {
        showError('Gagal menyalin naskah. Coba pilih teks secara manual.');
    }
});

// ── Download Text ──────────────────────────────

async function downloadDocx(text, filename) {
    try {
        const response = await fetch('/export-docx', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text, filename })
        });
        
        if (!response.ok) throw new Error('Gagal mengunduh dokumen.');
        
        const blob = await response.blob();
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `${filename}.docx`;
        a.click();
        URL.revokeObjectURL(url);
    } catch (err) {
        showError(err.message);
    }
}

// ── Loading State ──────────────────────────────

function setLoading(loading) {
    if (loading) {
        btnText.textContent = 'Memproses Audio...';
        btnSpinner.classList.remove('hidden');
        transcribeBtn.disabled = true;
        dropZone.style.pointerEvents = 'none';
        dropZone.style.opacity = '0.6';
    } else {
        btnText.textContent = 'Transkripsi Sekarang';
        btnSpinner.classList.add('hidden');
        transcribeBtn.disabled = !selectedFile;
        dropZone.style.pointerEvents = '';
        dropZone.style.opacity = '';
    }
}

// ── Utilities ──────────────────────────────────

function formatFileSize(bytes) {
    if (bytes < 1024) return bytes + ' B';
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
    return (bytes / (1024 * 1024)).toFixed(2) + ' MB';
}
