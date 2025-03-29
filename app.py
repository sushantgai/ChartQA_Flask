from flask import Flask, render_template, request, jsonify, url_for, session, send_file, Response
import torch
from PIL import Image
import pandas as pd
import re
import os
import base64
import json
import traceback
from io import BytesIO
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration

app = Flask(__name__)
app.secret_key = os.urandom(24)  # Required for session
UPLOAD_FOLDER = 'static/uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'} # Add allowed extensions

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Load PaliGemma model and processor (load once)
def load_paligemma_model():
    try:
        print("Loading PaliGemma model from local path...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

        # Specify the local path relative to your project structure
        local_model_path = os.path.join(os.path.dirname(__file__), 'Model')  # Update this path

        # Load model and processor from the specified local path
        model = PaliGemmaForConditionalGeneration.from_pretrained(
            local_model_path,
            torch_dtype=torch.float16
        )
        processor = AutoProcessor.from_pretrained(local_model_path)
        model = model.to(device)
        print("Model loaded successfully")
        return model, processor, device
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        traceback.print_exc()
        raise


# Store the model in the app context
with app.app_context():
    app.paligemma_model, app.paligemma_processor, app.device = load_paligemma_model()

# Helper function to check allowed extensions
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Clean model output function - improved like the Streamlit version
def clean_model_output(text):
    if not text:
        print("Warning: Empty text passed to clean_model_output")
        return ""
    
    # Check if the entire response is a print statement and extract its content
    print_match = re.search(r'^print\(["\'](.+?)["\']\)$', text.strip())
    if print_match:
        return print_match.group(1)
    
    # Remove all print statements
    text = re.sub(r'print\(.+?\)', '', text, flags=re.DOTALL)
    
    # Remove Python code formatting artifacts
    text = re.sub(r'```python|```', '', text)
    
    return text.strip()

# Analyze chart function
def analyze_chart_with_paligemma(image, query, use_cot=False):
    try:
        print(f"Starting analysis with query: {query}")
        print(f"Use CoT: {use_cot}")
        
        model = app.paligemma_model
        processor = app.paligemma_processor
        device = app.device
        
        # Add program of thought prefix if CoT is enabled (matching Streamlit version)
        if use_cot and not query.startswith("program of thought:"):
            modified_query = f"program of thought: {query}"
        else:
            modified_query = query
            
        print(f"Modified query: {modified_query}")
        
        # Process inputs
        try:
            print("Processing inputs...")
            inputs = processor(text=modified_query, images=image, return_tensors="pt")
            print(f"Input keys: {inputs.keys()}")
            prompt_length = inputs['input_ids'].shape[1]  # Store prompt length for later use
            inputs = {k: v.to(device) for k, v in inputs.items()}
        except Exception as e:
            print(f"Error processing inputs: {str(e)}")
            traceback.print_exc()
            return f"Error processing inputs: {str(e)}"

        # Generate output
        try:
            print("Generating output...")
            with torch.no_grad():
                generate_ids = model.generate(
                    **inputs,
                    num_beams=4,
                    max_new_tokens=512,
                    output_scores=True,
                    return_dict_in_generate=True
                )
            
            output_text = processor.batch_decode(
                generate_ids.sequences[:, prompt_length:],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )[0]
            
            print(f"Raw output text: {output_text}")
            cleaned_output = clean_model_output(output_text)
            print(f"Cleaned output text: {cleaned_output}")
            return cleaned_output
        except Exception as e:
            print(f"Error generating output: {str(e)}")
            traceback.print_exc()
            return f"Error generating output: {str(e)}"
            
    except Exception as e:
        print(f"Error in analyze_chart_with_paligemma: {str(e)}")
        traceback.print_exc()
        return f"Error: {str(e)}"

# Extract data points function - updated to match Streamlit version
def extract_data_points(image):
    print("Starting data extraction...")
    try:
        # Special query to extract data points - same as Streamlit
        extraction_query = "program of thought: Extract all data points from this chart. List each category or series and all its corresponding values in a structured format."
        
        print(f"Using extraction query: {extraction_query}")
        result = analyze_chart_with_paligemma(image, extraction_query, use_cot=True)
        print(f"Extraction result: {result}")
        
        # Parse the result into a DataFrame using the improved parser
        df = parse_chart_data(result)
        return df
    except Exception as e:
        print(f"Error extracting data points: {str(e)}")
        traceback.print_exc()
        return pd.DataFrame({'Error': [str(e)]})

# Parse chart data function - completely revamped to match Streamlit's implementation
def parse_chart_data(text):
    try:
        # Clean the text from print statements first
        text = clean_model_output(text)
        print(f"Parsing cleaned text: {text}")

        data = {}
        lines = text.split('\n')
        current_category = None

        # First pass: Look for category and value pairs
        for line in lines:
            if not line.strip():
                continue

            if ':' in line and not re.search(r'\d+\.\d+', line):
                current_category = line.split(':')[0].strip()
                data[current_category] = []
            elif current_category and (re.search(r'\d+', line) or ',' in line):
                value_match = re.findall(r'[-+]?\d*\.\d+|\d+', line)
                if value_match:
                    data[current_category].extend(value_match)

        # Second pass: If no categories found, try alternative pattern matching
        if not data:
            table_pattern = r'(\w+(?:\s\w+)*)\s*[:|]\s*((?:\d+(?:\.\d+)?(?:\s*,\s*\d+(?:\.\d+)?)*)|(?:\d+(?:\.\d+)?))'
            matches = re.findall(table_pattern, text)
            for category, values in matches:
                category = category.strip()
                if category not in data:
                    data[category] = []
                if ',' in values:
                    values = [v.strip() for v in values.split(',')]
                else:
                    values = [values.strip()]
                data[category].extend(values)

        # Convert all values to float where possible
        for key in data:
            data[key] = [float(val) if re.match(r'^[-+]?\d*\.?\d+$', val) else val for val in data[key]]

        # Create DataFrame
        if data:
            df = pd.DataFrame(data)
            print(f"Successfully parsed data: {df.head()}")
        else:
            df = pd.DataFrame({'Extracted_Text': [text]})
            print("Could not extract structured data, returning raw text")

        return df
    except Exception as e:
        print(f"Error parsing chart data: {str(e)}")
        traceback.print_exc()
        return pd.DataFrame({'Raw_Text': [text]})

@app.route('/')
def index():
    image_url = session.get('image_url', None)
    return render_template('index.html', image_url=image_url)

@app.route('/upload', methods=['POST'])
def upload_image():
    try:
        if 'image' not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files['image']
        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400

        if not allowed_file(file.filename):
             return jsonify({"error": "Invalid file type"}), 400

        filename = file.filename
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        session['image_url'] = url_for('static', filename=f'uploads/{filename}')
        session['image_filename'] = filename
        print(f"Image uploaded: {filename}")

        return jsonify({"image_url": session['image_url']})

    except Exception as e:
        print(f"Error in upload_image: {str(e)}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/analyze', methods=['POST'])
def analyze_chart():
    try:
        query = request.form['query']
        use_cot = request.form.get('use_cot') == 'true'
        image_filename = session.get('image_filename')
        
        if not image_filename:
            return jsonify({"error": "No image found in session. Please upload an image first."}), 400
            
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], image_filename)

        if not os.path.exists(image_path):
            return jsonify({"error": "Image not found. Please upload again."}), 400

        image = Image.open(image_path).convert('RGB')
        answer = analyze_chart_with_paligemma(image, query, use_cot)

        return jsonify({"answer": answer})

    except Exception as e:
        print(f"Error in analyze_chart: {str(e)}")
        traceback.print_exc()
        return jsonify({"error": str(e)})

@app.route('/extract', methods=['POST'])
def extract_data():
    try:
        image_filename = session.get('image_filename')
        
        if not image_filename:
            return jsonify({"error": "No image found in session. Please upload an image first."}), 400
            
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], image_filename)

        if not os.path.exists(image_path):
            return jsonify({"error": "Image not found. Please upload again."}), 400

        image = Image.open(image_path).convert('RGB')
        df = extract_data_points(image)
        
        # Check if DataFrame is empty or contains only error messages
        if df.empty:
            return jsonify({"error": "Could not extract data from the image"}), 400

        # Convert DataFrame to CSV data
        csv_data = df.to_csv(index=False)
        print(f"CSV data generated: {csv_data[:100]}...")  # Print first 100 chars

        # Encode CSV data to base64
        csv_base64 = base64.b64encode(csv_data.encode()).decode('utf-8')

        return jsonify({"csv_data": csv_base64})

    except Exception as e:
        print(f"Error in extract_data: {str(e)}")
        traceback.print_exc()
        return jsonify({"error": str(e)})

@app.route('/download_csv')
def download_csv():
    try:
        print("Download CSV route called")
        image_filename = session.get('image_filename')
        
        if not image_filename:
            print("No image in session")
            return jsonify({"error": "No image found in session. Please upload an image first."}), 400
            
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], image_filename)
        print(f"Looking for image at: {image_path}")

        if not os.path.exists(image_path):
            print("Image file not found")
            return jsonify({"error": "Image not found. Please upload again."}), 400

        print("Loading image")
        image = Image.open(image_path).convert('RGB')
        print("Extracting data points")
        df = extract_data_points(image)
        
        print(f"DataFrame: {df}")
        
        # Create a BytesIO object to hold the CSV data in memory
        csv_buffer = BytesIO()
        df.to_csv(csv_buffer, index=False, encoding='utf-8')
        csv_buffer.seek(0)  # Reset the buffer's position to the beginning
        
        # Debug: print CSV content
        csv_content = csv_buffer.getvalue().decode('utf-8')
        print(f"CSV Content: {csv_content}")
        csv_buffer.seek(0)  # Reset buffer position again after reading
        
        print("Preparing response")
        # Create direct response with CSV data
        response = Response(
            csv_buffer.getvalue(),
            mimetype='text/csv',
            headers={
                'Content-Disposition': 'attachment; filename=extracted_data.csv',
                'Content-Type': 'text/csv'
            }
        )
        
        print("Returning CSV response")
        return response

    except Exception as e:
        print(f"Error in download_csv: {str(e)}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

# Create a utility function to match the Streamlit version
def get_csv_download_link(df, filename="chart_data.csv"):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download CSV File</a>'
    return href

if __name__ == '__main__':
    app.run(debug=True)