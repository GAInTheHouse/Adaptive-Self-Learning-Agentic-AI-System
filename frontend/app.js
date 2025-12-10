// ==================== CONFIGURATION ====================
const API_BASE_URL = window.location.origin;
let selectedFile = null;
let currentPage = 0;
const PAGE_SIZE = 20;

// ==================== INITIALIZATION ====================
document.addEventListener('DOMContentLoaded', () => {
    initializeTabs();
    initializeTranscriptionMode();
    checkSystemHealth();
    loadDashboard();
    
    // Auto-refresh every 30 seconds
    setInterval(checkSystemHealth, 30000);
});

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
            break;
        case 'models':
            loadModelInfo();
            loadDeployedModel();
            refreshModelVersions();
            break;
        case 'monitoring':
            refreshPerformanceMetrics();
            refreshTrends();
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
    await Promise.all([
        refreshAgentStats(),
        refreshDataStats(),
        loadModelInfo()
    ]);
}

async function refreshAgentStats() {
    try {
        const response = await fetch(`${API_BASE_URL}/api/agent/stats`);
        const data = await response.json();
        
        const container = document.getElementById('agent-stats');
        const html = `
            <div class="stat-row">
                <span class="stat-label">Error Threshold</span>
                <span class="stat-value">${data.error_detection.threshold}</span>
            </div>
            <div class="stat-row">
                <span class="stat-label">Total Errors Detected</span>
                <span class="stat-value">${data.error_detection.total_errors_detected}</span>
            </div>
            <div class="stat-row">
                <span class="stat-label">Corrections Made</span>
                <span class="stat-value">${data.learning.corrections_made}</span>
            </div>
            <div class="stat-row">
                <span class="stat-label">Feedback Count</span>
                <span class="stat-value">${data.learning.feedback_count}</span>
            </div>
        `;
        
        container.innerHTML = html;
    } catch (error) {
        showToast('Failed to load agent statistics', 'error');
    }
}

async function refreshDataStats() {
    try {
        const response = await fetch(`${API_BASE_URL}/api/data/statistics`);
        const data = await response.json();
        
        const container = document.getElementById('data-stats');
        const correctionRate = (data.correction_rate * 100).toFixed(1);
        
        const html = `
            <div class="stat-row">
                <span class="stat-label">Total Failed Cases</span>
                <span class="stat-value">${data.total_failed_cases}</span>
            </div>
            <div class="stat-row">
                <span class="stat-label">Corrected Cases</span>
                <span class="stat-value">${data.corrected_cases}</span>
            </div>
            <div class="stat-row">
                <span class="stat-label">Correction Rate</span>
                <span class="stat-value">
                    <span class="badge ${correctionRate > 70 ? 'badge-success' : 'badge-warning'}">
                        ${correctionRate}%
                    </span>
                </span>
            </div>
            <div class="stat-row">
                <span class="stat-label">Average Error Score</span>
                <span class="stat-value">${data.average_error_score.toFixed(2)}</span>
            </div>
        `;
        
        container.innerHTML = html;
    } catch (error) {
        showToast('Failed to load data statistics', 'error');
    }
}

async function loadModelInfo() {
    try {
        const response = await fetch(`${API_BASE_URL}/api/models/info`);
        const data = await response.json();
        
        const container = document.getElementById('model-info');
        const html = `
            <div class="stat-row">
                <span class="stat-label">Model Name</span>
                <span class="stat-value">${data.name}</span>
            </div>
            <div class="stat-row">
                <span class="stat-label">Parameters</span>
                <span class="stat-value">${data.parameters.toLocaleString()}</span>
            </div>
            <div class="stat-row">
                <span class="stat-label">Device</span>
                <span class="stat-value">
                    <span class="badge ${data.device === 'cuda' ? 'badge-success' : 'badge-info'}">
                        ${data.device.toUpperCase()}
                    </span>
                </span>
            </div>
            <div class="stat-row">
                <span class="stat-label">Trainable Params</span>
                <span class="stat-value">${data.trainable_params.toLocaleString()}</span>
            </div>
        `;
        
        container.innerHTML = html;
        
        // Also update current model info in models tab
        document.getElementById('current-model-info').innerHTML = html;
    } catch (error) {
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
                        <span class="stat-label">Error Count</span>
                        <span class="stat-value">${detection.error_count || 0}</span>
                    </div>
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
        
        if (data.cases.length === 0) {
            container.innerHTML = '<p class="text-muted text-center">No failed cases found</p>';
            return;
        }
        
        const html = data.cases.map(caseItem => `
            <div class="case-item" onclick="showCaseDetails('${caseItem.case_id}')">
                <div class="case-header">
                    <span class="case-id">${caseItem.case_id}</span>
                    <span class="badge ${caseItem.corrected_transcript ? 'badge-success' : 'badge-warning'}">
                        ${caseItem.corrected_transcript ? 'Corrected' : 'Uncorrected'}
                    </span>
                </div>
                <div class="case-transcript">${caseItem.original_transcript.substring(0, 100)}...</div>
                <div class="case-meta">
                    <span><i class="fas fa-clock"></i> ${new Date(caseItem.timestamp).toLocaleString()}</span>
                    <span><i class="fas fa-exclamation-triangle"></i> Error Score: ${caseItem.error_score.toFixed(2)}</span>
                </div>
            </div>
        `).join('');
        
        container.innerHTML = html;
        
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
            <div class="result-section">
                <h4><i class="fas fa-list"></i> Error Types</h4>
                <div class="case-meta">
                    ${caseData.error_types.map(type => `<span class="badge badge-danger">${type}</span>`).join(' ')}
                </div>
            </div>
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
    try {
        const response = await fetch(`${API_BASE_URL}/api/data/datasets`);
        const data = await response.json();
        
        const container = document.getElementById('datasets-list');
        
        if (data.datasets.length === 0) {
            container.innerHTML = '<p class="text-muted">No datasets available</p>';
            return;
        }
        
        const html = data.datasets.map(dataset => `
            <div class="stat-row">
                <span class="stat-label">${dataset.dataset_id || dataset}</span>
                <span class="badge badge-info">Available</span>
            </div>
        `).join('');
        
        container.innerHTML = html;
    } catch (error) {
        console.error('Failed to load datasets:', error);
    }
}

// ==================== FINE-TUNING ====================
async function refreshFinetuningStatus() {
    try {
        const response = await fetch(`${API_BASE_URL}/api/finetuning/status`);
        
        if (response.status === 503) {
            document.getElementById('finetuning-status').innerHTML = 
                '<p class="text-muted">Fine-tuning coordinator not available</p>';
            return;
        }
        
        const data = await response.json();
        
        const container = document.getElementById('finetuning-status');
        const html = `
            <div class="stat-row">
                <span class="stat-label">Status</span>
                <span class="stat-value">
                    <span class="badge badge-success">Operational</span>
                </span>
            </div>
            <div class="stat-row">
                <span class="stat-label">Ready for Fine-tuning</span>
                <span class="stat-value">
                    <span class="badge ${data.orchestrator?.ready_for_finetuning ? 'badge-success' : 'badge-warning'}">
                        ${data.orchestrator?.ready_for_finetuning ? 'Yes' : 'No'}
                    </span>
                </span>
            </div>
            <div class="stat-row">
                <span class="stat-label">Total Jobs</span>
                <span class="stat-value">${data.orchestrator?.job_count || 0}</span>
            </div>
        `;
        
        container.innerHTML = html;
    } catch (error) {
        document.getElementById('finetuning-status').innerHTML = 
            '<p class="text-muted">Failed to load status</p>';
    }
}

async function triggerFinetuning() {
    const force = document.getElementById('force-trigger').checked;
    
    if (!confirm('Are you sure you want to trigger fine-tuning?')) {
        return;
    }
    
    try {
        const response = await fetch(`${API_BASE_URL}/api/finetuning/trigger?force=${force}`, {
            method: 'POST'
        });
        
        if (response.status === 503) {
            showToast('Fine-tuning coordinator not available', 'error');
            return;
        }
        
        const result = await response.json();
        
        if (result.status === 'triggered') {
            showToast(`Fine-tuning job triggered: ${result.job_id}`, 'success');
            refreshJobs();
        } else {
            showToast('Conditions not met for fine-tuning', 'warning');
        }
    } catch (error) {
        showToast('Failed to trigger fine-tuning: ' + error.message, 'error');
    }
}

async function refreshJobs() {
    try {
        const response = await fetch(`${API_BASE_URL}/api/finetuning/jobs`);
        
        if (response.status === 503) {
            document.getElementById('jobs-list').innerHTML = 
                '<p class="text-muted">Fine-tuning coordinator not available</p>';
            return;
        }
        
        const data = await response.json();
        const container = document.getElementById('jobs-list');
        
        if (data.jobs.length === 0) {
            container.innerHTML = '<p class="text-muted">No fine-tuning jobs yet</p>';
            return;
        }
        
        const html = data.jobs.map(job => `
            <div class="case-item">
                <div class="case-header">
                    <span class="case-id">${job.job_id}</span>
                    <span class="badge badge-info">${job.status}</span>
                </div>
                <div class="case-meta">
                    <span><i class="fas fa-clock"></i> ${new Date(job.created_at).toLocaleString()}</span>
                    ${job.dataset_id ? `<span><i class="fas fa-database"></i> ${job.dataset_id}</span>` : ''}
                </div>
            </div>
        `).join('');
        
        container.innerHTML = html;
    } catch (error) {
        document.getElementById('jobs-list').innerHTML = 
            '<p class="text-muted">Failed to load jobs</p>';
    }
}

// ==================== MODELS ====================
async function loadDeployedModel() {
    try {
        const response = await fetch(`${API_BASE_URL}/api/models/deployed`);
        
        if (response.status === 503) {
            document.getElementById('deployed-model-info').innerHTML = 
                '<p class="text-muted">Model management not available</p>';
            return;
        }
        
        const data = await response.json();
        const container = document.getElementById('deployed-model-info');
        
        if (!data.deployed) {
            container.innerHTML = '<p class="text-muted">No model deployed</p>';
            return;
        }
        
        const html = `
            <div class="stat-row">
                <span class="stat-label">Version ID</span>
                <span class="stat-value">${data.deployed.version_id}</span>
            </div>
            <div class="stat-row">
                <span class="stat-label">Model Name</span>
                <span class="stat-value">${data.deployed.model_name}</span>
            </div>
            <div class="stat-row">
                <span class="stat-label">Deployed At</span>
                <span class="stat-value">${new Date(data.deployed.created_at).toLocaleString()}</span>
            </div>
        `;
        
        container.innerHTML = html;
    } catch (error) {
        document.getElementById('deployed-model-info').innerHTML = 
            '<p class="text-muted">Failed to load deployed model</p>';
    }
}

async function refreshModelVersions() {
    try {
        const response = await fetch(`${API_BASE_URL}/api/models/versions`);
        
        if (response.status === 503) {
            document.getElementById('model-versions-list').innerHTML = 
                '<p class="text-muted">Model management not available</p>';
            return;
        }
        
        const data = await response.json();
        const container = document.getElementById('model-versions-list');
        
        if (data.versions.length === 0) {
            container.innerHTML = '<p class="text-muted">No model versions registered</p>';
            return;
        }
        
        const html = data.versions.map(version => `
            <div class="case-item">
                <div class="case-header">
                    <span class="case-id">${version.version_id}</span>
                    <span class="badge ${version.status === 'deployed' ? 'badge-success' : 'badge-info'}">
                        ${version.status}
                    </span>
                </div>
                <div class="case-meta">
                    <span><i class="fas fa-cube"></i> ${version.model_name}</span>
                    <span><i class="fas fa-clock"></i> ${new Date(version.created_at).toLocaleString()}</span>
                </div>
            </div>
        `).join('');
        
        container.innerHTML = html;
    } catch (error) {
        document.getElementById('model-versions-list').innerHTML = 
            '<p class="text-muted">Failed to load model versions</p>';
    }
}

// ==================== MONITORING ====================
async function refreshPerformanceMetrics() {
    try {
        const response = await fetch(`${API_BASE_URL}/api/metadata/performance`);
        const data = await response.json();
        
        const container = document.getElementById('performance-metrics');
        const html = `
            <div class="stat-row">
                <span class="stat-label">Total Inferences</span>
                <span class="stat-value">${data.overall_stats?.total_inferences || 0}</span>
            </div>
            <div class="stat-row">
                <span class="stat-label">Average Inference Time</span>
                <span class="stat-value">${(data.overall_stats?.avg_inference_time || 0).toFixed(2)}s</span>
            </div>
            <div class="stat-row">
                <span class="stat-label">Error Detection Rate</span>
                <span class="stat-value">${((data.overall_stats?.error_rate || 0) * 100).toFixed(1)}%</span>
            </div>
            <div class="stat-row">
                <span class="stat-label">Correction Rate</span>
                <span class="stat-value">${((data.overall_stats?.correction_rate || 0) * 100).toFixed(1)}%</span>
            </div>
        `;
        
        container.innerHTML = html;
    } catch (error) {
        document.getElementById('performance-metrics').innerHTML = 
            '<p class="text-muted">Failed to load performance metrics</p>';
    }
}

async function refreshTrends() {
    const metric = document.getElementById('trend-metric').value;
    const days = parseInt(document.getElementById('trend-days').value);
    
    try {
        const response = await fetch(`${API_BASE_URL}/api/metadata/trends?metric=${metric}&days=${days}`);
        const data = await response.json();
        
        const container = document.getElementById('trends-chart');
        
        if (!data.trend || data.trend.length === 0) {
            container.innerHTML = '<p class="text-muted">No trend data available</p>';
            return;
        }
        
        // Simple text-based trend display (you could integrate Chart.js for visual charts)
        const html = `
            <div class="stat-row">
                <span class="stat-label">Metric</span>
                <span class="stat-value">${metric.toUpperCase()}</span>
            </div>
            <div class="stat-row">
                <span class="stat-label">Data Points</span>
                <span class="stat-value">${data.trend.length}</span>
            </div>
            <p class="text-muted mt-10">
                Trend data available. Integrate Chart.js or similar library for visual representation.
            </p>
        `;
        
        container.innerHTML = html;
    } catch (error) {
        document.getElementById('trends-chart').innerHTML = 
            '<p class="text-muted">Failed to load trend data</p>';
    }
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

