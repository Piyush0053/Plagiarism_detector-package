/**
 * FROMI Plagiarism Detector - Main JavaScript
 * Handles client-side interactions for the plagiarism detection web application
 */

document.addEventListener('DOMContentLoaded', function() {
    // Initialize file upload functionality
    initializeFileUploads();
    
    // Initialize text area character counters
    initializeTextAreaCounters();
    
    // Initialize smooth scrolling for anchor links
    initializeSmoothScrolling();
    
    // Initialize tab functionality
    initializeTabs();
    
    // Initialize tooltips
    initializeTooltips();
    
    // Initialize loading indicators for forms
    initializeLoadingIndicators();
});

/**
 * Initialize file upload functionality
 * - Handle drag and drop file uploads
 * - Show selected filenames
 * - Handle file input changes
 */
function initializeFileUploads() {
    // Get all file upload areas
    const uploadAreas = document.querySelectorAll('.upload-area');
    
    uploadAreas.forEach(area => {
        const fileInput = area.querySelector('.file-input');
        const fileLabel = area.querySelector('.file-label');
        
        if (!fileInput || !fileLabel) return;
        
        // Handle drag events
        area.addEventListener('dragover', (e) => {
            e.preventDefault();
            area.classList.add('dragging');
        });
        
        area.addEventListener('dragleave', () => {
            area.classList.remove('dragging');
        });
        
        area.addEventListener('drop', (e) => {
            e.preventDefault();
            area.classList.remove('dragging');
            
            if (e.dataTransfer.files.length) {
                fileInput.files = e.dataTransfer.files;
                updateFileLabel(fileInput, fileLabel);
            }
        });
        
        // Handle click to select file
        area.addEventListener('click', () => {
            fileInput.click();
        });
        
        // Update label when file is selected
        fileInput.addEventListener('change', () => {
            updateFileLabel(fileInput, fileLabel);
        });
    });
}

/**
 * Update file label with selected filename(s)
 * @param {HTMLInputElement} fileInput - The file input element
 * @param {HTMLElement} fileLabel - The element to display the filename
 */
function updateFileLabel(fileInput, fileLabel) {
    if (fileInput.files.length === 0) {
        fileLabel.textContent = 'No file selected';
        fileLabel.classList.add('text-muted');
        return;
    }
    
    fileLabel.classList.remove('text-muted');
    
    if (fileInput.files.length === 1) {
        fileLabel.textContent = fileInput.files[0].name;
    } else {
        fileLabel.textContent = `${fileInput.files.length} files selected`;
    }
}

/**
 * Initialize text area character counters
 * Shows the current character count in text areas
 */
function initializeTextAreaCounters() {
    const textAreas = document.querySelectorAll('textarea[data-count]');
    
    textAreas.forEach(textArea => {
        const counterId = textArea.dataset.count;
        const counter = document.getElementById(counterId);
        
        if (!counter) return;
        
        // Initial count
        updateCharacterCount(textArea, counter);
        
        // Update count on input
        textArea.addEventListener('input', () => {
            updateCharacterCount(textArea, counter);
        });
    });
}

/**
 * Update character count for a text area
 * @param {HTMLTextAreaElement} textArea - The text area element
 * @param {HTMLElement} counter - The counter element
 */
function updateCharacterCount(textArea, counter) {
    const count = textArea.value.length;
    counter.textContent = `${count} characters`;
}

/**
 * Initialize smooth scrolling for anchor links
 */
function initializeSmoothScrolling() {
    const anchors = document.querySelectorAll('a[href^="#"]');
    
    anchors.forEach(anchor => {
        anchor.addEventListener('click', function(e) {
            const targetId = this.getAttribute('href');
            if (targetId === '#') return;
            
            const targetElement = document.querySelector(targetId);
            if (!targetElement) return;
            
            e.preventDefault();
            
            window.scrollTo({
                top: targetElement.offsetTop - 80,
                behavior: 'smooth'
            });
        });
    });
}

/**
 * Initialize tabs functionality
 */
function initializeTabs() {
    const tabs = document.querySelectorAll('[data-bs-toggle="tab"]');
    
    tabs.forEach(tab => {
        tab.addEventListener('click', function(e) {
            e.preventDefault();
            this.tab = new bootstrap.Tab(this);
            this.tab.show();
        });
    });
}

/**
 * Initialize Bootstrap tooltips
 */
function initializeTooltips() {
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    tooltipTriggerList.map(function(tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });
}

/**
 * Initialize loading indicators for forms
 * Shows a loading spinner when a form is submitted
 */
function initializeLoadingIndicators() {
    const forms = document.querySelectorAll('form');
    
    forms.forEach(form => {
        form.addEventListener('submit', function() {
            const submitBtn = this.querySelector('button[type="submit"]');
            if (!submitBtn) return;
            
            // Disable submit button and show loading indicator
            submitBtn.disabled = true;
            submitBtn.innerHTML = '<span class="spinner-border spinner-border-sm me-2" role="status" aria-hidden="true"></span> Processing...';
        });
    });
} 