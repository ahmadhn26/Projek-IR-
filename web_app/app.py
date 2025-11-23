"""
Flask Web Application untuk Sentiment Analysis Transjakarta
"""

from flask import Flask, render_template, request, jsonify
from model_loader import model_loader
import os

app = Flask(__name__)

# Load models saat startup
print("\n" + "="*70)
print("INITIALIZING SENTIMENT ANALYSIS WEB APP")
print("="*70 + "\n")

try:
    model_loader.load_all()
    print("✓ Application ready!\n")
except Exception as e:
    print(f"✗ Error loading models: {e}")
    print("Please run 'python train_and_save_models.py' first!\n")


# ==========================================
# ROUTES - Pages
# ==========================================

@app.route('/')
def index():
    """Landing page"""
    stats = model_loader.get_dataset_stats()
    models_info = model_loader.get_all_models_info()
    
    # Get best model
    best_model = None
    if models_info:
        best_model = max(models_info, key=lambda x: x.get('Test_F1_Macro', 0))
    
    return render_template('index.html', 
                         stats=stats, 
                         best_model=best_model)


@app.route('/dashboard')
def dashboard():
    """Dashboard dengan visualisasi"""
    stats = model_loader.get_dataset_stats()
    models_info = model_loader.get_all_models_info()
    
    return render_template('dashboard.html',
                         stats=stats,
                         models_info=models_info)


@app.route('/analyzer')
def analyzer():
    """Real-time sentiment analyzer"""
    models = list(model_loader.models.keys())
    return render_template('analyzer.html', models=models)


@app.route('/comparison')
def comparison():
    """Model comparison page"""
    models_info = model_loader.get_all_models_info()
    return render_template('comparison.html', models_info=models_info)


@app.route('/methodology')
def methodology():
    """Methodology page"""
    return render_template('methodology.html')


# ==========================================
# API ENDPOINTS
# ==========================================

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """
    API endpoint untuk prediksi sentiment.
    
    Request JSON:
    {
        "text": "input text",
        "model": "SVM" (optional, default: "SVM")
    }
    
    Response JSON:
    {
        "success": true,
        "sentiment": "Positif" or "Negatif",
        "confidence": 0.95,
        "probabilities": {"Negatif": 0.05, "Positif": 0.95},
        "preprocessed_text": "...",
        "preprocessing_steps": [...]
    }
    """
    try:
        data = request.get_json()
        
        if not data or 'text' not in data:
            return jsonify({
                'success': False,
                'error': 'No text provided'
            }), 400
        
        text = data['text']
        model_name = data.get('model', 'SVM')
        
        # Validate text
        if not text or len(text.strip()) == 0:
            return jsonify({
                'success': False,
                'error': 'Text is empty'
            }), 400
        
        # Predict
        result = model_loader.predict(text, model_name)
        
        return jsonify({
            'success': True,
            **result
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/stats')
def api_stats():
    """
    API endpoint untuk dataset statistics.
    
    Response JSON:
    {
        "total_samples": 2757,
        "negative_samples": 1500,
        "positive_samples": 1257,
        ...
    }
    """
    try:
        stats = model_loader.get_dataset_stats()
        return jsonify(stats)
    except Exception as e:
        return jsonify({
            'error': str(e)
        }), 500


@app.route('/api/models')
def api_models():
    """
    API endpoint untuk model information.
    
    Response JSON:
    [
        {
            "Model": "SVM",
            "Test_Accuracy": 0.86,
            "Test_F1_Macro": 0.85,
            ...
        },
        ...
    ]
    """
    try:
        models_info = model_loader.get_all_models_info()
        return jsonify(models_info)
    except Exception as e:
        return jsonify({
            'error': str(e)
        }), 500


# ==========================================
# ERROR HANDLERS
# ==========================================

@app.errorhandler(404)
def not_found(e):
    """404 error handler"""
    return render_template('404.html'), 404


@app.errorhandler(500)
def server_error(e):
    """500 error handler"""
    return render_template('500.html'), 500


# ==========================================
# MAIN
# ==========================================

if __name__ == '__main__':
    print("="*70)
    print("Starting Flask server...")
    print("Open your browser and go to: http://localhost:5000")
    print("Press CTRL+C to stop the server")
    print("="*70 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
