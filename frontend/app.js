// ==================== CONFIGURATION ====================
const API_BASE_URL = window.location.origin;
let selectedFile = null;
let currentPage = 0;
const PAGE_SIZE = 20;
let performanceMock = null;

// ==================== INITIALIZATION ====================
document.addEventListener('DOMContentLoaded', () => {
    initializeTabs();
    initializeTranscriptionMode();
    initializeModelSelector();
    checkSystemHealth();
    loadDashboard();
    
    // Auto-refresh every 30 seconds
    setInterval(checkSystemHealth, 30000);
});

// Initialize model selector by loading available models from backend
async function initializeModelSelector() {
    const modelSelector = document.getElementById('model-selector');
    if (!modelSelector) return;
    
    try {
        const response = await fetch(`${API_BASE_URL}/api/models/available`);
        if (!response.ok) {
            throw new Error('Failed to fetch available models');
        }
        
        const data = await response.json();
        const models = data.models || [];
        const defaultModel = data.default || 'wav2vec2-base';
        
        // Clear existing options
        modelSelector.innerHTML = '';
        
        // Add options for each available model
        models.forEach(model => {
            if (model.is_available) {
                const option = document.createElement('option');
                option.value = model.id;  // Use the actual model identifier
                option.textContent = model.display_name || model.name;
                if (model.id === defaultModel || model.is_current) {
                    option.selected = true;
                }
                modelSelector.appendChild(option);
            }
        });
        
        // If no models found, add a fallback
        if (modelSelector.options.length === 0) {
            const option = document.createElement('option');
            option.value = 'wav2vec2-base';
            option.textContent = 'Wav2Vec2 Base';
            option.selected = true;
            modelSelector.appendChild(option);
        }
    } catch (e) {
        console.error('Could not initialize model selector:', e);
        // Fallback to default options
        modelSelector.innerHTML = `
            <option value="wav2vec2-base">Wav2Vec2 Base</option>
            <option value="wav2vec2-finetuned">Fine-tuned Wav2Vec2</option>
        `;
    }
}

// ==================== TAB NAVIGATION ====================
function initializeTabs() {
    const tabButtons = document.querySelectorAll('.tab-btn');
    
    tabButtons.forEach(button => {
        button.addEventListener('click', () => {
            const tabName = button.dataset.tab;
            switchTab(tabName);
        });
    });
}

function switchTab(tabName) {
    // Clear any existing auto-refresh intervals when switching tabs
    if (window.finetuningRefreshInterval) {
        clearInterval(window.finetuningRefreshInterval);
        window.finetuningRefreshInterval = null;
    }
    
    // Clear any job polling intervals
    if (window.jobPollInterval) {
        clearInterval(window.jobPollInterval);
        window.jobPollInterval = null;
    }
    
    // Update buttons
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.classList.remove('active');
    });
    document.querySelector(`[data-tab="${tabName}"]`).classList.add('active');
    
    // Update content
    document.querySelectorAll('.tab-content').forEach(content => {
        content.classList.remove('active');
    });
    document.getElementById(tabName).classList.add('active');
    
    // Load tab-specific data
    loadTabData(tabName);
}

function loadTabData(tabName) {
    switch(tabName) {
        case 'dashboard':
            loadDashboard();
            break;
        case 'data':
            loadFailedCases();
            refreshDataStats();
            refreshDatasets();
            break;
        case 'finetuning':
            refreshFinetuningStatus();
            refreshJobs();
            // Auto-refresh fine-tuning status every 5 seconds when tab is active
            clearInterval(window.finetuningRefreshInterval);
            window.finetuningRefreshInterval = setInterval(() => {
                if (document.getElementById('finetuning').classList.contains('active')) {
                    refreshFinetuningStatus();
                    refreshJobs();
                }
            }, 5000);
            break;
        case 'models':
            loadModelInfo();
            refreshModelVersions();
            break;
        case 'monitoring':
            refreshPerformanceMetrics();
            refreshTrends();
            break;
        case 'transcribe':
            // Ensure model selector is set to fine-tuned if available
            initializeModelSelector();
            break;
    }
}

// ==================== SYSTEM HEALTH ====================
async function checkSystemHealth() {
    try {
        const response = await fetch(`${API_BASE_URL}/api/health`);
        const data = await response.json();
        
        const statusElement = document.getElementById('system-status');
        statusElement.classList.remove('offline');
        statusElement.classList.add('online');
        statusElement.querySelector('span').textContent = 'System Online';
        
        updateHealthDisplay(data);
    } catch (error) {
        const statusElement = document.getElementById('system-status');
        statusElement.classList.remove('online');
        statusElement.classList.add('offline');
        statusElement.querySelector('span').textContent = 'System Offline';
    }
}

function updateHealthDisplay(health) {
    const container = document.getElementById('health-info');
    
    performanceMock = {
        total_inferences: health.components?.agent?.total_inferences || 0,
        avg_inference_time: 0.0,
        avg_error_score: 0.0
    };
    
    const html = `
        <div class="stat-row">
            <span class="stat-label">Baseline Model</span>
            <span class="stat-value">${health.components.baseline_model.model} (${health.components.baseline_model.device})</span>
        </div>
        <div class="stat-row">
            <span class="stat-label">Agent Status</span>
            <span class="stat-value">
                <span class="badge badge-success">Operational</span>
            </span>
        </div>
        <div class="stat-row">
            <span class="stat-label">LLM Available</span>
            <span class="stat-value">
                ${health.components.agent.llm_available ? 
                    '<span class="badge badge-success">Yes</span>' : 
                    '<span class="badge badge-warning">No</span>'}
            </span>
        </div>
        <div class="stat-row">
            <span class="stat-label">Last Check</span>
            <span class="stat-value">${new Date().toLocaleTimeString()}</span>
        </div>
    `;
    
    container.innerHTML = html;
}

// ==================== DASHBOARD ====================
async function loadDashboard() {
    // Dashboard now shows health and current model info only
    await Promise.all([
        loadModelInfo()
    ]);
}

async function loadModelInfo() {
    const container = document.getElementById('model-info');
    if (!container) return;
    
    try {
        // Get current model (will return fine-tuned if available, else base)
        const response = await fetch(`${API_BASE_URL}/api/models/info`);
        const data = await response.json();
        
        // Get WER/CER from the same response (no separate API call needed)
        const wer = data.wer;
        const cer = data.cer;
        
        // Build WER/CER display
        let metricsHtml = '';
        if (wer !== null && wer !== undefined && cer !== null && cer !== undefined) {
            metricsHtml = `
                <div class="stat-row">
                    <span class="stat-label">WER</span>
                    <span class="stat-value">${(wer * 100).toFixed(2)}%</span>
                </div>
                <div class="stat-row">
                    <span class="stat-label">CER</span>
                    <span class="stat-value">${(cer * 100).toFixed(2)}%</span>
                </div>
            `;
        } else {
            metricsHtml = `
                <div class="stat-row">
                    <span class="stat-label">Performance</span>
                    <span class="stat-value">Not Evaluated</span>
                </div>
            `;
        }
        
        const html = `
            <div class="stat-row">
                <span class="stat-label">Model Name</span>
                <span class="stat-value">${data.name}</span>
            </div>
            ${metricsHtml}
            <div class="stat-row">
                <span class="stat-label">Parameters</span>
                <span class="stat-value">${data.parameters.toLocaleString()}</span>
            </div>
            <div class="stat-row">
                <span class="stat-label">Trainable Params</span>
                <span class="stat-value">${data.trainable_params.toLocaleString()}</span>
            </div>
            <div class="stat-row">
                <span class="stat-label">Device</span>
                <span class="stat-value">${data.device}</span>
            </div>
        `;
        
        container.innerHTML = html;
        
        // Also update current model info in models tab
        const modelsTabContainer = document.getElementById('current-model-info');
        if (modelsTabContainer) {
            modelsTabContainer.innerHTML = html;
        }
    } catch (error) {
        console.error('Error loading model information:', error);
        container.innerHTML = '<p class="text-danger">Failed to load model information.</p>';
        showToast('Failed to load model information', 'error');
    }
}

// ==================== TRANSCRIPTION ====================
function initializeTranscriptionMode() {
    const modeInputs = document.querySelectorAll('input[name="transcribe-mode"]');
    const agentOptions = document.getElementById('agent-options');
    
    modeInputs.forEach(input => {
        input.addEventListener('change', (e) => {
            if (e.target.value === 'agent') {
                agentOptions.style.display = 'block';
            } else {
                agentOptions.style.display = 'none';
            }
        });
    });
}

function handleFileSelect(event) {
    const file = event.target.files[0];
    if (!file) return;
    
    selectedFile = file;
    
    const fileInfo = document.getElementById('file-info');
    fileInfo.classList.remove('hidden');
    fileInfo.innerHTML = `
        <i class="fas fa-file-audio"></i>
        <div>
            <strong>${file.name}</strong><br>
            <small>${(file.size / 1024 / 1024).toFixed(2)} MB</small>
        </div>
    `;
    
    // Audio preview
    const audioEl = document.getElementById('audio-preview');
    if (audioEl) {
        const blobUrl = URL.createObjectURL(file);
        audioEl.src = blobUrl;
        audioEl.classList.remove('hidden');
        audioEl.load();
    }
    
    document.getElementById('transcribe-btn').disabled = false;
}

async function transcribeAudio() {
    if (!selectedFile) {
        showToast('Please select an audio file', 'warning');
        return;
    }
    
    const mode = document.querySelector('input[name="transcribe-mode"]:checked').value;
    const selectedModel = document.getElementById('model-selector').value;
    const autoCorrection = document.getElementById('auto-correction')?.checked || false;
    const recordErrors = document.getElementById('record-errors')?.checked || false;
    
    const transcribeBtn = document.getElementById('transcribe-btn');
    const originalText = transcribeBtn.innerHTML;
    transcribeBtn.disabled = true;
    
    // Show different loading messages based on mode
    if (mode === 'agent') {
        transcribeBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Transcribing with STT...';
    } else {
        transcribeBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Transcribing...';
    }
    
    // Show loading state in transcript boxes
    const sttBox = document.getElementById('stt-original-transcript');
    const llmBox = document.getElementById('llm-refined-transcript');
    
    if (sttBox) {
        sttBox.innerHTML = '<p class="text-muted"><i class="fas fa-spinner fa-spin"></i> STT processing...</p>';
    }
    
    if (llmBox) {
        if (mode === 'agent') {
            llmBox.innerHTML = '<p class="text-muted"><i class="fas fa-spinner fa-spin"></i> LLM is analyzing and refining transcript... (this may take 10-15 seconds)</p>';
        } else {
            llmBox.innerHTML = '<p class="text-muted">No LLM correction in baseline mode</p>';
        }
    }
    
    // Show the result container early so user sees loading state
    const resultContainer = document.getElementById('transcription-result');
    resultContainer.classList.remove('hidden');
    
    try {
        const formData = new FormData();
        formData.append('file', selectedFile);
        
        let url = `${API_BASE_URL}/api/transcribe/${mode}`;
        const params = new URLSearchParams();
        params.append('model', selectedModel);
        if (mode === 'agent') {
            params.append('auto_correction', autoCorrection);
            params.append('record_if_error', recordErrors);
        }
        url += `?${params.toString()}`;
        
        const response = await fetch(url, {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            throw new Error('Transcription failed');
        }
        
        const result = await response.json();
        displayTranscriptionResult(result, mode, selectedModel);
        showToast('Transcription completed successfully', 'success');
    } catch (error) {
        showToast('Transcription failed: ' + error.message, 'error');
        document.getElementById('stt-original-transcript').innerHTML = '<p class="text-muted text-danger">Error: ' + error.message + '</p>';
        document.getElementById('llm-refined-transcript').innerHTML = '<p class="text-muted text-danger">Error occurred</p>';
    } finally {
        transcribeBtn.disabled = false;
        transcribeBtn.innerHTML = originalText;
    }
}

function displayTranscriptionResult(result, mode, selectedModel) {
    const container = document.getElementById('transcription-result');
    container.classList.remove('hidden');
    
    // Get transcripts - use original_transcript for STT and transcript (or corrected) for LLM refined
    const sttOriginal = result.original_transcript || result.transcript || 'No transcription available';
    
    // Update the side-by-side transcript display
    const sttBox = document.getElementById('stt-original-transcript');
    const llmBox = document.getElementById('llm-refined-transcript');
    
    if (sttBox) {
        sttBox.innerHTML = `<p>${sttOriginal}</p>`;
    }
    
    if (llmBox) {
        if (mode === 'baseline') {
            // Baseline mode: no LLM correction, show same as STT
            llmBox.innerHTML = `<p class="text-muted">No LLM correction in baseline mode. Use Agent mode to see LLM-refined transcript.</p>`;
        } else {
            // Agent mode: show LLM refined transcript
            const llmRefined = result.corrected_transcript || result.transcript || 'No refined transcription available';
            llmBox.innerHTML = `<p>${llmRefined}</p>`;
        }
    }
    
    // Remove any existing additional info sections (except transcripts-comparison)
    const existingSections = container.querySelectorAll('.result-section');
    existingSections.forEach(section => {
        if (!section.closest('.transcripts-comparison')) {
            section.remove();
        }
    });
    
    // Build additional info section
    let html = `
        <div class="result-section">
            <h4><i class="fas fa-info-circle"></i> Model Information</h4>
            <div class="stat-row">
                <span class="stat-label">Selected Model</span>
                <span class="stat-value">${selectedModel}</span>
            </div>
            <div class="stat-row">
                <span class="stat-label">Mode</span>
                <span class="stat-value">${mode === 'agent' ? 'Agent (with LLM correction)' : 'Baseline'}</span>
            </div>
        </div>
    `;
    
    if (mode === 'agent' && result.error_detection) {
        const detection = result.error_detection;
        html += `
            <div class="result-section">
                <h4><i class="fas fa-exclamation-triangle"></i> Error Detection</h4>
                <div class="stat-row">
                    <span class="stat-label">Has Errors</span>
                    <span class="stat-value">
                        <span class="badge ${detection.has_errors ? 'badge-danger' : 'badge-success'}">
                            ${detection.has_errors ? 'Yes' : 'No'}
                        </span>
                    </span>
                </div>
            ${detection.has_errors ? `
                <div class="stat-row">
                    <span class="stat-label">Error Score</span>
                    <span class="stat-value">${(detection.error_score || 0).toFixed(2)}</span>
                </div>
            ` : `
                    <div class="stat-row">
                        <span class="stat-label">Status</span>
                        <span class="stat-value">No errors detected - model performing well!</span>
                    </div>
                `}
            </div>
        `;
        
        if (result.corrections && result.corrections.applied) {
            html += `
                <div class="result-section">
                    <h4><i class="fas fa-check-circle"></i> Corrections Applied</h4>
                    <div class="stat-row">
                        <span class="stat-label">Correction Count</span>
                        <span class="stat-value">${result.corrections.count || 0}</span>
                    </div>
                </div>
            `;
        }
        
        if (result.case_id) {
            html += `
                <div class="result-section">
                    <h4><i class="fas fa-save"></i> Case Recorded</h4>
                    <p class="text-muted">Case ID: <code>${result.case_id}</code></p>
                    <p class="text-muted">This error case will be used for fine-tuning the model.</p>
                </div>
            `;
        }
    }
    
    html += `
        <div class="result-section">
            <h4><i class="fas fa-clock"></i> Performance</h4>
            <div class="stat-row">
                <span class="stat-label">Inference Time</span>
                <span class="stat-value">${(result.inference_time_seconds || 0).toFixed(2)}s</span>
            </div>
        </div>
    `;
    
    // Append additional info after the transcripts comparison
    const comparisonSection = container.querySelector('.transcripts-comparison');
    if (comparisonSection) {
        comparisonSection.insertAdjacentHTML('afterend', html);
    } else {
        container.insertAdjacentHTML('beforeend', html);
    }
    
    container.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

// ==================== DATA MANAGEMENT ====================
async function loadFailedCases(pageDirection = 0) {
    if (pageDirection === 0) {
        currentPage = 0;
    } else if (pageDirection === 1) {
        currentPage++;
    } else {
        currentPage = Math.max(0, currentPage - 1);
    }
    
    try {
        const offset = currentPage * PAGE_SIZE;
        const response = await fetch(`${API_BASE_URL}/api/data/failed-cases?limit=${PAGE_SIZE}&offset=${offset}`);
        const data = await response.json();
        
        const container = document.getElementById('failed-cases-list');
        
        if (!data.cases || data.cases.length === 0) {
            container.innerHTML = '<p class="text-muted text-center">No failed cases found</p>';
        } else {
            const html = data.cases.map(caseItem => `
                <div class="case-item" onclick="showCaseDetails('${caseItem.case_id}')">
                    <div class="case-header">
                        <span class="case-id">${caseItem.case_id}</span>
                    </div>
                    <div class="case-transcript">${(caseItem.original_transcript || '').substring(0, 120)}...</div>
                    <div class="case-meta">
                        <span><i class="fas fa-clock"></i> ${new Date(caseItem.timestamp).toLocaleString()}</span>
                        <span><i class="fas fa-exclamation-triangle"></i> Error Score: ${caseItem.error_score.toFixed(2)}</span>
                    </div>
                </div>
            `).join('');
            
            container.innerHTML = html;
        }
        
        // Update pagination
        document.getElementById('page-info').textContent = `Page ${currentPage + 1}`;
        document.getElementById('prev-btn').disabled = currentPage === 0;
        document.getElementById('next-btn').disabled = data.cases.length < PAGE_SIZE;
    } catch (error) {
        showToast('Failed to load failed cases', 'error');
    }
}

async function showCaseDetails(caseId) {
    try {
        const response = await fetch(`${API_BASE_URL}/api/data/case/${caseId}`);
        const caseData = await response.json();
        
        const modal = document.getElementById('case-modal');
        const modalBody = document.getElementById('modal-body');
        
        const html = `
            <div class="result-section">
                <h4><i class="fas fa-info-circle"></i> Case Information</h4>
                <div class="stat-row">
                    <span class="stat-label">Case ID</span>
                    <span class="stat-value"><code>${caseData.case_id}</code></span>
                </div>
                <div class="stat-row">
                    <span class="stat-label">Timestamp</span>
                    <span class="stat-value">${new Date(caseData.timestamp).toLocaleString()}</span>
                </div>
                <div class="stat-row">
                    <span class="stat-label">Error Score</span>
                    <span class="stat-value">${caseData.error_score.toFixed(2)}</span>
                </div>
            </div>
            <div class="result-section">
                <h4><i class="fas fa-file-alt"></i> Original Transcript</h4>
                <div class="transcript-box">${caseData.original_transcript}</div>
            </div>
            ${caseData.corrected_transcript ? `
                <div class="result-section">
                    <h4><i class="fas fa-check-circle"></i> Corrected Transcript</h4>
                    <div class="transcript-box">${caseData.corrected_transcript}</div>
                </div>
            ` : `
                <div class="result-section">
                    <h4><i class="fas fa-edit"></i> Add Correction</h4>
                    <textarea id="correction-input" class="input" rows="4" placeholder="Enter corrected transcript..."></textarea>
                    <button class="btn btn-primary mt-10" onclick="submitCorrection('${caseData.case_id}')">
                        <i class="fas fa-save"></i> Save Correction
                    </button>
                </div>
            `}
        `;
        
        modalBody.innerHTML = html;
        modal.classList.remove('hidden');
    } catch (error) {
        showToast('Failed to load case details', 'error');
    }
}

async function submitCorrection(caseId) {
    const correctionText = document.getElementById('correction-input').value.trim();
    if (!correctionText) {
        showToast('Please enter a correction', 'warning');
        return;
    }
    try {
        const response = await fetch(`${API_BASE_URL}/api/data/correction`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                case_id: caseId,
                corrected_transcript: correctionText,
                correction_method: 'manual'
            })
        });
        
        if (!response.ok) {
            throw new Error('Failed to submit correction');
        }
        
        showToast('Correction saved successfully', 'success');
        closeModal();
        loadFailedCases();
    } catch (error) {
        showToast('Failed to save correction: ' + error.message, 'error');
    }
}

async function prepareDataset() {
    const minErrorScore = parseFloat(document.getElementById('min-error-score').value);
    const maxSamples = parseInt(document.getElementById('max-samples').value);
    const balanceErrors = document.getElementById('balance-errors').checked;
    const createVersion = document.getElementById('create-version').checked;
    
    try {
        const response = await fetch(`${API_BASE_URL}/api/data/prepare-dataset`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                min_error_score: minErrorScore,
                max_samples: maxSamples,
                balance_error_types: balanceErrors,
                create_version: createVersion
            })
        });
        
        if (!response.ok) {
            throw new Error('Dataset preparation failed');
        }
        
        const result = await response.json();
        showToast(`Dataset prepared successfully! Dataset ID: ${result.dataset_id}`, 'success');
        refreshDatasets();
    } catch (error) {
        showToast('Failed to prepare dataset: ' + error.message, 'error');
    }
}

async function refreshDatasets() {
    const container = document.getElementById('datasets-list');
    container.innerHTML = '<div class="loading">Loading...</div>';
    const samplePath = 'data/sample_recordings_for_UI/';
    try {
        const response = await fetch(`${API_BASE_URL}/api/data/sample-recordings`);
        if (!response.ok) throw new Error('Failed to load sample recordings');
        const data = await response.json();
        if (!data.files || data.files.length === 0) {
            container.innerHTML = `
                <p class="text-muted">
                    No files found. Add audio files to <code>${samplePath}</code> and click refresh.
                </p>
            `;
            return;
        }
        const html = data.files.map(file => `
            <div class="stat-row">
                <span class="stat-label">${file.name}</span>
                <span class="stat-value text-muted">${file.path}</span>
            </div>
        `).join('');
        container.innerHTML = html;
    } catch (error) {
        container.innerHTML = `
            <p class="text-muted">
                Failed to list files. Ensure the server can read <code>${samplePath}</code>.
            </p>
        `;
    }
}

// ==================== FINE-TUNING ====================
async function refreshFinetuningStatus() {
    const container = document.getElementById('finetuning-status');
    try {
        // Add cache-busting timestamp to ensure fresh data
        const timestamp = new Date().getTime();
        const response = await fetch(`${API_BASE_URL}/api/finetuning/status?t=${timestamp}`, {
            cache: 'no-cache',
            headers: {
                'Cache-Control': 'no-cache'
            }
        });
        if (!response.ok) {
            throw new Error('Failed to fetch status');
        }
        const data = await response.json();
        
        const orchestrator = data.orchestrator || {};
        const status = data.status || 'unknown';
        const errorCount = orchestrator.error_cases_count || 0;
        const totalJobs = orchestrator.total_jobs || 0;
        const activeJobs = orchestrator.active_jobs || 0;
        const minErrorCases = orchestrator.min_error_cases || 100;
        const casesNeeded = orchestrator.cases_needed || 0;
        const casesNeededMessage = orchestrator.cases_needed_message || '';
        const shouldTrigger = orchestrator.should_trigger || false;
        
        // Determine status badge color
        let statusBadgeClass = 'badge-secondary';
        if (status === 'ready' || status === 'operational') {
            statusBadgeClass = 'badge-success';
        } else if (status === 'active') {
            statusBadgeClass = 'badge-info';
        } else if (status === 'unavailable') {
            statusBadgeClass = 'badge-secondary';
        } else if (status === 'error') {
            statusBadgeClass = 'badge-danger';
        } else {
            statusBadgeClass = 'badge-warning';
        }
        
        const html = `
            <div class="stat-row">
                <span class="stat-label">Status</span>
                <span class="stat-value">
                    <span class="badge ${statusBadgeClass}">${status}</span>
                </span>
            </div>
            <div class="stat-row">
                <span class="stat-label">Error Cases</span>
                <span class="stat-value">${errorCount} / ${minErrorCases}</span>
            </div>
            <div class="stat-row">
                <span class="stat-label">Threshold</span>
                <span class="stat-value">${minErrorCases} cases minimum</span>
            </div>
            ${casesNeeded > 0 ? `
            <div class="stat-row" style="margin-top: 10px; padding-top: 10px; border-top: 1px solid #e0e0e0;">
                <span class="stat-label" style="color: #f39c12;">
                    <i class="fas fa-info-circle"></i> Status
                </span>
                <span class="stat-value" style="color: #f39c12; font-weight: 500;">
                    ${casesNeededMessage}
                </span>
            </div>
            ` : shouldTrigger ? `
            <div class="stat-row" style="margin-top: 10px; padding-top: 10px; border-top: 1px solid #e0e0e0;">
                <span class="stat-label" style="color: #27ae60;">
                    <i class="fas fa-check-circle"></i> Status
                </span>
                <span class="stat-value" style="color: #27ae60; font-weight: 500;">
                    Ready to trigger fine-tuning
                </span>
            </div>
            ` : ''}
            <div class="stat-row">
                <span class="stat-label">Total Jobs</span>
                <span class="stat-value">${totalJobs}</span>
            </div>
            <div class="stat-row">
                <span class="stat-label">Active Jobs</span>
                <span class="stat-value">${activeJobs}</span>
            </div>
        `;
        
        container.innerHTML = html;
    } catch (error) {
        container.innerHTML = `
            <div class="stat-row">
                <span class="stat-label">Status</span>
                <span class="stat-value">
                    <span class="badge badge-secondary">Unavailable</span>
                </span>
            </div>
            <div class="text-muted">${error.message}</div>
        `;
    }
}

async function triggerFinetuning() {
    const force = document.getElementById('force-trigger').checked;
    
    if (!confirm('Are you sure you want to trigger fine-tuning?')) {
        return;
    }
    
    // Show "Running Fine-Tuning" message
    const statusMessageDiv = document.getElementById('finetuning-status-message');
    const statusText = document.getElementById('finetuning-status-text');
    statusMessageDiv.style.display = 'block';
    statusText.innerHTML = '<span class="badge badge-info">Running Fine-Tuning...</span>';
    
    // Disable the trigger button
    const triggerBtn = document.querySelector('button[onclick="triggerFinetuning()"]');
    const originalBtnText = triggerBtn.innerHTML;
    triggerBtn.disabled = true;
    triggerBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Processing...';
    
    let jobId = null;
    let pollCount = 0;
    const maxPollAttempts = 60; // Poll for up to 5 minutes (60 * 5 seconds)
    
    try {
        const response = await fetch(`${API_BASE_URL}/api/finetuning/trigger?force=${force}`, {
            method: 'POST'
        });
        
        if (response.status === 503) {
            statusMessageDiv.style.display = 'none';
            triggerBtn.disabled = false;
            triggerBtn.innerHTML = originalBtnText;
            showToast('Fine-tuning coordinator not available', 'error');
            return;
        }
        
        const result = await response.json();
        
        if (result.status === 'triggered' || result.status === 'not_triggered') {
            if (result.status === 'triggered') {
                jobId = result.job_id;
                showToast(`Fine-tuning job triggered: ${jobId}`, 'success');
                
                // Start polling for the job to appear in the list
                // Store in window so we can clean it up if needed
                window.jobPollInterval = setInterval(async () => {
                    pollCount++;
                    
                    try {
                        // Refresh jobs list
                        await refreshJobs();
                        
                        // Check if job appears in the list
                        const jobsResponse = await fetch(`${API_BASE_URL}/api/finetuning/jobs`);
                        if (jobsResponse.ok) {
                            const jobsData = await jobsResponse.json();
                            const jobs = jobsData.jobs || [];
                            const job = jobs.find(j => j.job_id === jobId);
                            
                            if (job) {
                                // Job found! Show "Finished" message
                                clearInterval(window.jobPollInterval);
                                window.jobPollInterval = null;
                                statusText.innerHTML = '<span class="badge badge-success">Finished</span>';
                                triggerBtn.disabled = false;
                                triggerBtn.innerHTML = originalBtnText;
                                
                                // Hide status message after 5 seconds
                                setTimeout(() => {
                                    statusMessageDiv.style.display = 'none';
                                }, 5000);
                                
                                // Refresh status
                                refreshFinetuningStatus();
                                return;
                            }
                        }
                        
                        // If we've exceeded max attempts, stop polling
                        if (pollCount >= maxPollAttempts) {
                            clearInterval(window.jobPollInterval);
                            window.jobPollInterval = null;
                            statusText.innerHTML = '<span class="badge badge-warning">Job may still be processing. Check jobs list.</span>';
                            triggerBtn.disabled = false;
                            triggerBtn.innerHTML = originalBtnText;
                            showToast('Job may still be processing. Please check the jobs list.', 'warning');
                        }
                    } catch (error) {
                        console.error('Error polling for job:', error);
                        // Continue polling on error
                    }
                }, 5000); // Poll every 5 seconds
            } else {
                statusMessageDiv.style.display = 'none';
                triggerBtn.disabled = false;
                triggerBtn.innerHTML = originalBtnText;
                showToast('Conditions not met for fine-tuning', 'warning');
            }
        } else {
            statusMessageDiv.style.display = 'none';
            triggerBtn.disabled = false;
            triggerBtn.innerHTML = originalBtnText;
            showToast('Unexpected response from server', 'error');
        }
    } catch (error) {
        if (window.jobPollInterval) {
            clearInterval(window.jobPollInterval);
            window.jobPollInterval = null;
        }
        statusMessageDiv.style.display = 'none';
        triggerBtn.disabled = false;
        triggerBtn.innerHTML = originalBtnText;
        showToast('Failed to trigger fine-tuning: ' + error.message, 'error');
    }
}

async function refreshJobs() {
    const container = document.getElementById('jobs-list');
    try {
        const response = await fetch(`${API_BASE_URL}/api/finetuning/jobs`);
        if (!response.ok) {
            throw new Error('Failed to fetch jobs');
        }
        const data = await response.json();
        const jobs = data.jobs || [];
        
        if (jobs.length === 0) {
            container.innerHTML = '<p class="text-muted text-center">No fine-tuning jobs found</p>';
            return;
        }
        
            const html = jobs.map(job => {
            const status = job.status || 'unknown';
            const jobId = job.job_id || 'N/A';
            const createdAt = job.created_at || job.created_at_timestamp || new Date().toISOString();
            const datasetId = job.dataset_id || job.config?.dataset_id || '';
            // Get model version from config (set during training), not from version_id
            const modelVersion = job.config?.model_version || '';
            const isCurrent = job.config?.is_current || false;
            
            // Map status to display status
            let displayStatus = status;
            let statusBadgeClass = 'badge-secondary';
            if (status === 'completed') {
                displayStatus = 'Completed';
                statusBadgeClass = 'badge-success';
            } else if (status === 'failed') {
                displayStatus = 'Failed';
                statusBadgeClass = 'badge-danger';
            } else if (status === 'training' || status === 'evaluating') {
                displayStatus = 'Running';
                statusBadgeClass = 'badge-info';
            } else if (status === 'preparing' || status === 'ready') {
                displayStatus = 'Preparing';
                statusBadgeClass = 'badge-warning';
            } else if (status === 'pending') {
                displayStatus = 'Started';
                statusBadgeClass = 'badge-info';
            } else {
                displayStatus = status.charAt(0).toUpperCase() + status.slice(1);
            }
            
            // Build model info for completed jobs
            let modelInfoHtml = '';
            if (status === 'completed' && modelVersion) {
                modelInfoHtml = `
                    <div class="case-meta" style="margin-top: 8px; padding-top: 8px; border-top: 1px solid #e0e0e0;">
                        <span><i class="fas fa-cube"></i> Model: <strong>${modelVersion}</strong></span>
                        ${isCurrent ? `<span class="badge badge-success" style="margin-left: 8px;"><i class="fas fa-check-circle"></i> Current Model</span>` : ''}
                    </div>
                `;
            }
            
            return `
                <div class="case-item">
                    <div class="case-header">
                        <span class="case-id">${jobId}</span>
                        <span class="badge ${statusBadgeClass}">${displayStatus}</span>
                    </div>
                    <div class="case-meta">
                        <span><i class="fas fa-clock"></i> ${new Date(createdAt).toLocaleString()}</span>
                        ${datasetId ? `<span><i class="fas fa-database"></i> ${datasetId}</span>` : ''}
                        ${job.trigger_reason ? `<span><i class="fas fa-info-circle"></i> ${job.trigger_reason}</span>` : ''}
                    </div>
                    ${modelInfoHtml}
                </div>
            `;
        }).join('');
        
        container.innerHTML = html;
    } catch (error) {
        container.innerHTML = `<p class="text-muted text-center">Failed to load jobs: ${error.message}</p>`;
    }
}

// ==================== MODELS ====================
async function refreshModelVersions() {
    const container = document.getElementById('model-versions-list');
    let versions = [];
    
    try {
        const response = await fetch(`${API_BASE_URL}/api/models/versions`);
        if (response.ok) {
            const data = await response.json();
            versions = data.versions || [];
        }
    } catch (e) {
        console.warn('Could not fetch model versions:', e);
        // Fallback to defaults
        versions = [
            {
                version_id: 'baseline',
                model_name: 'Wav2Vec2 Base',
                is_current: false,
                created_at: null,
                parameters: 95000000
            }
        ];
    }
    
    // If no versions found, show baseline
    if (versions.length === 0) {
        versions = [
            {
                version_id: 'baseline',
                model_name: 'Wav2Vec2 Base',
                is_current: false,
                created_at: null,
                parameters: 95000000
            }
        ];
    }
    
    const html = versions.map(version => {
        const isCurrent = version.is_current !== undefined ? version.is_current : (version.status === 'current');
        const isBaseline = version.version_id === 'wav2vec2-base' || version.model_id === 'wav2vec2-base';
        // Display WER/CER instead of parameters
        const wer = version.wer !== null && version.wer !== undefined ? `${(version.wer * 100).toFixed(1)}%` : 'N/A';
        const cer = version.cer !== null && version.cer !== undefined ? `${(version.cer * 100).toFixed(1)}%` : 'N/A';
        const metrics = version.is_finetuned !== false ? `WER: ${wer} / CER: ${cer}` : 'N/A';
        const createdDate = version.created_at ? new Date(version.created_at).toLocaleString() : 'N/A';
        
        // Determine badge text and class - only show badges for baseline and current models
        let badgeHtml = '';
        if (isBaseline) {
            badgeHtml = `<span class="badge badge-secondary">Baseline</span>`;
        } else if (isCurrent) {
            badgeHtml = `<span class="badge badge-success">Current</span>`;
        }
        // No badge for intermediate models (neither baseline nor current)
        
        return `
        <div class="case-item">
            <div class="case-header">
                <span class="case-id">${version.version_id}</span>
                ${badgeHtml}
            </div>
            <div class="case-meta">
                <span><i class="fas fa-cube"></i> ${version.model_name || version.version_id}</span>
                <span><i class="fas fa-chart-line"></i> ${metrics}</span>
                <span><i class="fas fa-clock"></i> ${createdDate}</span>
            </div>
        </div>
    `;
    }).join('');
    
    container.innerHTML = html;
}

// ==================== MONITORING ====================
async function refreshPerformanceMetrics() {
    const container = document.getElementById('performance-metrics');
    let data;
    try {
        const response = await fetch(`${API_BASE_URL}/api/metadata/performance`);
        if (response.ok) {
            data = await response.json();
        }
    } catch (e) {
        // ignore, fallback to defaults
    }
    
    // Get evaluation results (WER/CER) from dedicated endpoint
    let evalData = { baseline: { wer: 0.36, cer: 0.13 }, finetuned: { wer: 0.36, cer: 0.13 } };
    try {
        const evalResponse = await fetch(`${API_BASE_URL}/api/models/evaluation`);
        if (evalResponse.ok) {
            evalData = await evalResponse.json();
        }
    } catch (e) {
        console.warn('Could not fetch evaluation results:', e);
    }
    
    const stats = data?.overall_stats || {};
    
    // Use evaluation results for baseline and current (fine-tuned) model
    const baselineWer = evalData.baseline?.wer ?? stats.baseline_wer ?? 0.36;
    const baselineCer = evalData.baseline?.cer ?? stats.baseline_cer ?? 0.13;
    const currentWer = evalData.finetuned?.wer ?? stats.finetuned_wer ?? baselineWer;
    const currentCer = evalData.finetuned?.cer ?? stats.finetuned_cer ?? baselineCer;
    
    performanceMock = {
        total_inferences: stats.total_inferences ?? 0,
        avg_inference_time: stats.avg_inference_time ?? 0.0,
        avg_error_score: stats.avg_error_score ?? 0.0,
        wer_baseline: baselineWer,
        wer_finetuned: currentWer,
        cer_baseline: baselineCer,
        cer_finetuned: currentCer
    };
    
    const html = `
        <div class="stat-row">
            <span class="stat-label">Total Inferences</span>
            <span class="stat-value">${performanceMock.total_inferences}</span>
        </div>
        <div class="stat-row">
            <span class="stat-label">Average Inference Time</span>
            <span class="stat-value">${performanceMock.avg_inference_time.toFixed(2)}s</span>
        </div>
        <div class="stat-row">
            <span class="stat-label">Average Error Score</span>
            <span class="stat-value">${performanceMock.avg_error_score.toFixed(3)}</span>
        </div>
        ${performanceMock.wer_baseline !== undefined ? `<div class="stat-row">
            <span class="stat-label">Baseline WER / CER</span>
            <span class="stat-value">${(performanceMock.wer_baseline * 100).toFixed(1)}% / ${((performanceMock.cer_baseline || 0) * 100).toFixed(2)}%</span>
        </div>` : ''}
        ${performanceMock.wer_finetuned !== undefined ? `<div class="stat-row">
            <span class="stat-label">Current Model WER / CER</span>
            <span class="stat-value">${(performanceMock.wer_finetuned * 100).toFixed(1)}% / ${((performanceMock.cer_finetuned || 0) * 100).toFixed(2)}%</span>
        </div>` : ''}
    `;
    
    container.innerHTML = html;
    refreshTrends();
}

async function refreshTrends() {
    const metric = document.getElementById('trend-metric').value;
    const days = parseInt(document.getElementById('trend-days').value);
    const container = document.getElementById('trends-chart');
    
    // Use performance data to build a two-point trend (baseline vs fine-tuned)
    // Only show if WER/CER data is available
    if (performanceMock?.wer_baseline === undefined && performanceMock?.cer_baseline === undefined) {
        container.innerHTML = '<p class="text-muted">WER/CER data not available</p>';
        return;
    }
    const baseVal = metric === 'wer' ? (performanceMock?.wer_baseline ?? 0) * 100 : (performanceMock?.cer_baseline ?? 0) * 100;
    const currentVal = metric === 'wer' ? (performanceMock?.wer_finetuned ?? 0) * 100 : (performanceMock?.cer_finetuned ?? 0) * 100;
    const points = [
        { label: 'Baseline', value: baseVal },
        { label: 'Current Model', value: currentVal }
    ];
    
    const html = `
        <div class="stat-row">
            <span class="stat-label">Metric</span>
            <span class="stat-value">${metric.toUpperCase()}</span>
        </div>
        <div class="stat-row">
            <span class="stat-label">Window</span>
            <span class="stat-value">Last ${days} days</span>
        </div>
        <div class="transcript-box" style="max-height:220px; overflow:auto;">
            ${points.map(p => `
                <div class="stat-row">
                    <span class="stat-label">${p.label}</span>
                    <span class="stat-value">${p.value.toFixed(2)}%</span>
                </div>
                <div style="height:8px; background:rgba(102,126,234,0.2); border-radius:4px; overflow:hidden; margin:6px 0 12px 0;">
                    <div style="height:8px; width:${Math.min(p.value, 100)}%; background:linear-gradient(135deg, #667eea, #764ba2);"></div>
                </div>
            `).join('')}
        </div>
    `;
    
    container.innerHTML = html;
}

// ==================== UTILITY FUNCTIONS ====================
function closeModal() {
    document.getElementById('case-modal').classList.add('hidden');
}

function showToast(message, type = 'info') {
    const container = document.getElementById('toast-container');
    const toast = document.createElement('div');
    toast.className = `toast toast-${type}`;
    
    const icon = {
        success: 'check-circle',
        error: 'exclamation-circle',
        warning: 'exclamation-triangle',
        info: 'info-circle'
    }[type] || 'info-circle';
    
    toast.innerHTML = `
        <i class="fas fa-${icon}"></i>
        <span>${message}</span>
    `;
    
    container.appendChild(toast);
    
    setTimeout(() => {
        toast.remove();
    }, 5000);
}

// Close modal when clicking outside
document.addEventListener('click', (e) => {
    const modal = document.getElementById('case-modal');
    if (e.target === modal) {
        closeModal();
    }
});

