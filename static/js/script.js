document.addEventListener('DOMContentLoaded', function() {
    const uploadForm = document.getElementById('uploadForm');
    const uploadStatus = document.getElementById('uploadStatus');
    const imageSection = document.getElementById('imageSection');
    const chartPreview = document.getElementById('chartPreview');
    const analysisSection = document.getElementById('analysisSection');
    const analysisResults = document.getElementById('analysisResults');
    const analyzeButton = document.getElementById('analyzeButton');
    const queryInput = document.getElementById('query');
    const useCotCheckbox = document.getElementById('use_cot');
    const extractSection = document.getElementById('extractSection');
    const extractButton = document.getElementById('extractButton');
    const downloadSection = document.getElementById('downloadSection');
    const downloadLink = document.getElementById('downloadLink');
    const extractStatus = document.getElementById('extractStatus');

    // Function to show a message
    function showMessage(element, message, isError = false) {
        element.textContent = message;
        element.style.color = isError ? 'red' : 'green';
    }

    // Function to clear a message
    function clearMessage(element) {
        element.textContent = '';
    }

    // Handle image upload
    uploadForm.addEventListener('submit', async function(event) {
        event.preventDefault();
        clearMessage(uploadStatus);

        const formData = new FormData(uploadForm);

        try {
            const response = await fetch('/upload', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();

            if (data.error) {
                showMessage(uploadStatus, data.error, true);
                imageSection.style.display = 'none';
                analysisSection.style.display = 'none';
                extractSection.style.display = 'none';
                downloadSection.style.display = 'none';

            } else {
                chartPreview.src = data.image_url;
                imageSection.style.display = 'block';
                analysisSection.style.display = 'block';
                extractSection.style.display = 'block';
                downloadSection.style.display = 'none'; // Hide initially
                showMessage(uploadStatus, 'Image uploaded successfully!');

            }
        } catch (error) {
            showMessage(uploadStatus, 'An error occurred during upload.', true);
            console.error('Upload error:', error);
            imageSection.style.display = 'none';
            analysisSection.style.display = 'none';
            extractSection.style.display = 'none';
            downloadSection.style.display = 'none';
        }
    });

    // Handle analyze chart
    analyzeButton.addEventListener('click', async function() {
        clearMessage(analysisResults);

        const query = queryInput.value;
        const useCot = useCotCheckbox.checked;

        if (!query) {
            showMessage(analysisResults, 'Please enter a question.', true);
            return;
        }

        const formData = new FormData();
        formData.append('query', query);
        formData.append('use_cot', useCot);

        try {
            const response = await fetch('/analyze', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();

            if (data.error) {
                showMessage(analysisResults, data.error, true);
            } else {
                analysisResults.textContent = 'Answer: ' + data.answer;
                analysisResults.style.color = 'black';
            }
        } catch (error) {
            showMessage(analysisResults, 'An error occurred during analysis.', true);
            console.error('Analysis error:', error);
        }
    });

     // Handle extract data
    extractButton.addEventListener('click', async function() {
        clearMessage(extractStatus);
        downloadSection.style.display = 'none'; // Hide until data is ready

        try {
            const response = await fetch('/extract', {
                method: 'POST'
            });

            const data = await response.json();

            if (data.error) {
                showMessage(extractStatus, data.error, true);
            } else {
                // CSV data is in base64 format
                const csvData = atob(data.csv_data); // Decode base64
                const blob = new Blob([csvData], { type: 'text/csv' });
                const url = URL.createObjectURL(blob);

                downloadLink.href = url;
                downloadLink.style.display = 'inline';  // Show the download link
                downloadSection.style.display = 'block';  // Show the whole section

                showMessage(extractStatus, 'Data extracted successfully!');
            }
        } catch (error) {
            showMessage(extractStatus, 'An error occurred during extraction.', true);
            console.error('Extraction error:', error);
        }
    });

});
