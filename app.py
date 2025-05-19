import os
import jwt
import random
import smtplib
import torch
import numpy as np
import cv2
import pickle
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import uuid
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime, timedelta
from functools import wraps
from flask import Flask, request, jsonify, render_template, redirect, url_for, session
from flask_pymongo import PyMongo
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from flask_wtf.csrf import CSRFProtect
from tensorflow.keras.models import load_model as keras_load_model

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY')
app.config['MONGO_URI'] = os.getenv('MONGO_URI')
app.config['JWT_SECRET_KEY'] = os.getenv('JWT_SECRET_KEY')
app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] = os.getenv('MAIL_USERNAME')
app.config['MAIL_PASSWORD'] = os.getenv('MAIL_PASSWORD')
app.config['MODEL_PATH'] = 'resnet_bird_drone_classifier.h5'

# Add configuration settings for the image classifier
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'pkl'}
app.config['KERAS_MODEL_PATH'] = 'resnet_bird_drone_classifier.h5'

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Add this before mongo = PyMongo(app)
if not app.config.get('MONGO_URI'):
    app.config['MONGO_URI'] = 'mongodb://localhost:27017/auth_app'

try:
    mongo = PyMongo(app)
    print("MongoDB connected successfully")
except Exception as e:
    print(f"Warning: MongoDB connection failed: {e}")
    # For development only - this allows the app to start without MongoDB
    from unittest.mock import MagicMock
    mongo = MagicMock()
    mongo.db.users = MagicMock()
    mongo.db.users.find_one = lambda x: None

# Initialize CSRF protection
csrf = CSRFProtect(app)

# Add HTTPS enforcement
@app.before_request
def force_https():
    # Check if we're already using HTTPS
    if request.headers.get('X-Forwarded-Proto') == 'http':
        url = request.url.replace('http://', 'https://', 1)
        return redirect(url, code=301)

def allowed_file(filename):
    """Check if uploaded file has an allowed extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def preprocess_image(file_path):
    """Preprocess uploaded image or spectrogram for model prediction."""
    try:
        if file_path.endswith(".pkl"):
            with open(file_path, "rb") as f:
                spectrogram = pickle.load(f)

            # Ensure spectrogram is real-valued (remove complex parts)
            spectrogram = np.abs(spectrogram)

            # Handle extra channels
            if len(spectrogram.shape) == 3:
                if spectrogram.shape[-1] > 3:
                    spectrogram = spectrogram[..., :3]  # Take first 3 channels
                elif spectrogram.shape[-1] == 1:
                    spectrogram = np.repeat(spectrogram, 3, axis=-1)  # Convert grayscale to RGB
        else:
            # Read standard image in color
            spectrogram = cv2.imread(file_path, cv2.IMREAD_COLOR)

        if spectrogram is None:
            raise ValueError("Error loading image. Ensure it's a valid format.")

        # Resize to ResNet input size
        spectrogram = cv2.resize(spectrogram, (224, 224))

        # Normalize image
        spectrogram = spectrogram / 255.0

        # Add batch dimension for model input
        spectrogram = np.expand_dims(spectrogram, axis=0)
        
        return spectrogram
    except Exception as e:
        print(f"Preprocessing Error: {e}")
        return None

def convert_pkl_to_image(pkl_path):
    """Convert a .pkl spectrogram file to a PNG image for display"""
    try:
        image_filename = f"{uuid.uuid4().hex}.png"
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], image_filename)
        
        with open(pkl_path, 'rb') as f:
            spectrogram = pickle.load(f)
        
        # Process spectrogram data for visualization
        spec_data = np.abs(spectrogram)
        
        if len(spec_data.shape) == 3:
            if spec_data.shape[-1] > 3:
                # Take first 3 channels if there are more
                spec_data = spec_data[..., :3]
                # Normalize for display
                spec_data = (spec_data - spec_data.min()) / (spec_data.max() - spec_data.min() + 1e-8)
            elif spec_data.shape[-1] == 1:
                spec_data = spec_data.squeeze()
        
        plt.figure(figsize=(8, 6))
        plt.imshow(spec_data, aspect='auto')
        plt.tight_layout()
        plt.axis('off')  # Remove axes
        plt.savefig(image_path, bbox_inches='tight', pad_inches=0)
        plt.close()
        
        return image_filename
    except Exception as e:
        print(f"Error converting .pkl to image: {e}")
        return None

# Update model loading function to prioritize your TensorFlow model
def load_model():
    """Load the appropriate model based on what's available"""
    try:
        # Try Keras/TF model first since we know it exists
        keras_model_path = app.config.get('KERAS_MODEL_PATH', 'resnet_bird_drone_classifier.h5')
        if os.path.exists(keras_model_path):
            model = keras_load_model(keras_model_path)
            print(f"Keras model loaded successfully from {keras_model_path}")
            return {'model': model, 'type': 'keras'}
        
        # Try PyTorch model as fallback (only if needed)
        pytorch_model_path = app.config['MODEL_PATH']
        if os.path.exists(pytorch_model_path):
            try:
                model = torch.load(pytorch_model_path)
                model.eval()
                print("PyTorch model loaded successfully")
                return {'model': model, 'type': 'pytorch'}
            except Exception as e:
                print(f"PyTorch model loading failed: {e}")
        
        raise ValueError(f"No valid model found. Looking for: {keras_model_path}")
    except Exception as e:
        print(f"Error loading any model: {e}")
        return None

# JWT token verification decorator
def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = None
        
        if 'x-access-token' in request.headers:
            token = request.headers['x-access-token']
        
        if not token:
            return jsonify({'message': 'Token is missing!'}), 401
        
        try:
            data = jwt.decode(token, app.config['JWT_SECRET_KEY'], algorithms=["HS256"])
            current_user = mongo.db.users.find_one({'email': data['email']})
        except:
            return jsonify({'message': 'Token is invalid!'}), 401
        
        return f(current_user, *args, **kwargs)
    
    return decorated

# Email OTP function
def send_otp_email(email, otp, is_registration=False, is_reset=False):
    msg = MIMEMultipart()
    msg['From'] = app.config['MAIL_USERNAME']
    msg['To'] = email
    
    if is_registration:
        msg['Subject'] = 'Verify Your Email Registration'
        body = f'Your OTP for email verification is: {otp}\n\nPlease enter this code to complete your registration.'
    elif is_reset:
        msg['Subject'] = 'Reset Your Password'
        body = f'Your OTP for password reset is: {otp}\n\nPlease enter this code to reset your password.'
    else:
        msg['Subject'] = 'Your OTP for Login'
        body = f'Your OTP for login is: {otp}'
    
    msg.attach(MIMEText(body, 'plain'))
    
    try:
        server = smtplib.SMTP(app.config['MAIL_SERVER'], app.config['MAIL_PORT'])
        server.starttls()
        server.login(app.config['MAIL_USERNAME'], app.config['MAIL_PASSWORD'])
        text = msg.as_string()
        server.sendmail(app.config['MAIL_USERNAME'], email, text)
        server.quit()
        return True
    except Exception as e:
        print(f"Error sending email: {e}")
        return False
# Routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        # Get form data
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        
        # Validate password
        if len(password) < 8:
            return render_template('register.html', error='Password must be at least 8 characters long')
        
        if not any(char.isupper() for char in password):
            return render_template('register.html', error='Password must contain at least one uppercase letter')
            
        if not any(char.isdigit() for char in password):
            return render_template('register.html', error='Password must contain at least one number')
            
        if not any(char in '!@#$%^&*()-_=+[]{}|;:,.<>?/~`' for char in password):
            return render_template('register.html', error='Password must contain at least one special character')
        
        # Check if user already exists
        existing_user = mongo.db.users.find_one({'email': email})
        if existing_user:
            return render_template('register.html', error='Email already registered')
        
        # Generate OTP
        otp = str(random.randint(100000, 999999))
        
        # Store registration data and OTP in session
        session['reg_username'] = username
        session['reg_email'] = email
        session['reg_password'] = password
        session['reg_otp'] = otp
        session['reg_otp_expiry'] = (datetime.now() + timedelta(minutes=10)).timestamp()
        
        # Send OTP via email
        if send_otp_email(email, otp, is_registration=True):
            return redirect(url_for('verify_registration_otp'))
        else:
            return render_template('register.html', error='Failed to send verification OTP')
    
    return render_template('register.html')

@app.route('/verify-registration-otp', methods=['GET', 'POST'])
def verify_registration_otp():
    if 'reg_otp' not in session or 'reg_email' not in session:
        return redirect(url_for('register'))
    
    if request.method == 'POST':
        user_otp = request.form.get('otp')
        
        if datetime.now().timestamp() > session.get('reg_otp_expiry', 0):
            # Clear session data
            for key in ['reg_username', 'reg_email', 'reg_password', 'reg_otp', 'reg_otp_expiry']:
                session.pop(key, None)
            return render_template('register.html', error='OTP expired. Please register again.')
        
        if user_otp == session['reg_otp']:
            # Create new user
            username = session['reg_username']
            email = session['reg_email']
            password = session['reg_password']
            
            hashed_password = generate_password_hash(password)
            user_data = {
                'username': username,
                'email': email,
                'password': hashed_password,
                'email_verified': True,
                'created_at': datetime.now()
            }
            
            mongo.db.users.insert_one(user_data)
            
            # Clear registration session data
            for key in ['reg_username', 'reg_email', 'reg_password', 'reg_otp', 'reg_otp_expiry']:
                session.pop(key, None)
            
            return redirect(url_for('login', message='Registration successful! Please login.'))
        
        return render_template('verify_registration_otp.html', error='Invalid OTP')
    
    return render_template('verify_registration_otp.html', email=session.get('reg_email'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    message = request.args.get('message', '')
    
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        
        user = mongo.db.users.find_one({'email': email})
        
        if not user:
            return render_template('login.html', error='Invalid email or password')
        
        # Check if the stored hash begins with 'scrypt:' which indicates an unsupported hash type
        if user['password'].startswith('scrypt:'):
            # Option 1: Reset the password (safer but requires user action)
            # Redirect to password reset flow
            session['reset_email'] = email
            return redirect(url_for('forgot_password', 
                                   message="Your account needs a password update. Please reset your password."))

            # Option 2: Bypass password check for migration (less secure, temporary fix)
            # Instead of the code above, you could use this code to migrate the user:
            # new_hash = generate_password_hash(password)
            # mongo.db.users.update_one({'email': email}, {'$set': {'password': new_hash}})
            # However, this assumes the user entered the correct password
        else:
            # Normal password check
            try:
                if check_password_hash(user['password'], password):
                    # Generate JWT token
                    token = jwt.encode({
                        'email': user['email'],
                        'exp': datetime.now() + timedelta(hours=24)
                    }, app.config['JWT_SECRET_KEY'])
                    
                    # Store token in session
                    session['token'] = token
                    session['email'] = email
                    
                    return redirect(url_for('model_page'))
            except ValueError as e:
                print(f"Password verification error: {e}")
                # If we hit an error with the password hash, guide the user to reset
                session['reset_email'] = email
                return redirect(url_for('forgot_password', 
                               message="Your password needs to be updated for security reasons. Please reset it."))
        
        # If we get here, password was wrong
        return render_template('login.html', error='Invalid email or password')
    
    return render_template('login.html', message=message)
@app.route('/model-page')
def model_page():
    if 'token' not in session:
        return redirect(url_for('login'))
    
    try:
        # Verify token
        data = jwt.decode(session['token'], app.config['JWT_SECRET_KEY'], algorithms=["HS256"])
        user = mongo.db.users.find_one({'email': data['email']})
        
        if not user:
            return redirect(url_for('login'))
        
        # Here you would load your model and render the model page
        return render_template('model_page.html', username=user['username'])
    
    except:
        return redirect(url_for('login'))

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

@app.route('/api/predict', methods=['POST'])
def predict():
    """Process file upload and make prediction."""
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file part"}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400

        if file and allowed_file(file.filename):
            # Save the uploaded file
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # For displaying to user: if it's a .pkl file, convert it to an image
            display_filename = filename
            if filename.endswith('.pkl'):
                image_filename = convert_pkl_to_image(filepath)
                if image_filename:
                    display_filename = image_filename
                else:
                    return jsonify({"error": "Failed to process spectrogram file"}), 400

            # Preprocess the file for model prediction
            preprocessed = preprocess_image(filepath)
            if preprocessed is None:
                return jsonify({"error": "Failed to preprocess image"}), 400

            # Load model and make prediction
            model_data = load_model()
            if model_data is None:
                return jsonify({"error": "Failed to load model"}), 500

            # Make prediction based on model type
            if model_data['type'] == 'keras':
                prediction = model_data['model'].predict(preprocessed)[0][0]
                # Classify as Bird or Drone based on threshold (0.35 from reference code)
                label = "Bird" if prediction >= 0.35 else "Drone"
                confidence = float(1.0 - prediction) if prediction >= 0.35 else float(prediction)
            else:  # PyTorch model
                with torch.no_grad():
                    tensor_input = torch.FloatTensor(preprocessed)
                    outputs = model_data['model'](tensor_input)
                    _, predicted = torch.max(outputs, 1)
                    label = "Bird" if predicted.item() == 0 else "Drone"
                    confidence = float(torch.nn.functional.softmax(outputs, dim=1)[0][predicted].item())

            # Return prediction in the format expected by the frontend
            return jsonify({
                "prediction": {
                    "class": label,
                    "confidence": confidence,
                    "display_image": f"/static/uploads/{display_filename}"
                }
            })

        return jsonify({"error": "File type not allowed"}), 400
    except Exception as e:
        import traceback
        print(f"Prediction error: {e}")
        print(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

@app.route('/resend-registration-otp')
def resend_registration_otp():
    if 'reg_email' not in session:
        return redirect(url_for('register'))
    
    # Generate new OTP
    if 'reg_email' not in session:
        return redirect(url_for('register'))
    
    # Generate new OTP
    otp = str(random.randint(100000, 999999))
    
    # Update session with new OTP
    session['reg_otp'] = otp
    session['reg_otp_expiry'] = (datetime.now() + timedelta(minutes=10)).timestamp()
    
    # Send OTP via email
    if send_otp_email(session['reg_email'], otp, is_registration=True):
        return redirect(url_for('verify_registration_otp', message='New OTP sent!'))
    else:
        return render_template('verify_registration_otp.html', 
                              error='Failed to send new OTP', 
                              email=session.get('reg_email'))
# Add these routes after the existing routes

@app.route('/verify-email', methods=['GET', 'POST'])
def verify_email():
    if request.method == 'POST':
        email = request.form.get('email')
        
        # Check if user exists
        user = mongo.db.users.find_one({'email': email})
        
        if not user:
            return render_template('verify_email.html', error='Email not found in our records')
            
        if user.get('email_verified', False):
            return redirect(url_for('login', message='Your email is already verified. Please login.'))
        
        # Generate OTP
        otp = str(random.randint(100000, 999999))
        
        # Store email verification data in session
        session['verify_email'] = email
        session['verify_otp'] = otp
        session['verify_otp_expiry'] = (datetime.now() + timedelta(minutes=10)).timestamp()
        
        # Send OTP via email
        if send_otp_email(email, otp, is_registration=True):
            return redirect(url_for('confirm_email'))
        else:
            return render_template('verify_email.html', error='Failed to send verification email')
    
    return render_template('verify_email.html')

@app.route('/confirm-email', methods=['GET', 'POST'])
def confirm_email():
    if 'verify_email' not in session or 'verify_otp' not in session:
        return redirect(url_for('verify_email'))
    
    email = session.get('verify_email')
    
    if request.method == 'POST':
        user_otp = request.form.get('otp')
        
        if datetime.now().timestamp() > session.get('verify_otp_expiry', 0):
            # Clear session data
            for key in ['verify_email', 'verify_otp', 'verify_otp_expiry']:
                session.pop(key, None)
            return render_template('verify_email.html', error='OTP expired. Please try again.')
        
        if user_otp == session['verify_otp']:
            # Update user's email verification status
            mongo.db.users.update_one({'email': email}, {'$set': {'email_verified': True}})
            
            # Clear verification session data
            for key in ['verify_email', 'verify_otp', 'verify_otp_expiry']:
                session.pop(key, None)
            
            return redirect(url_for('login', message='Email verified successfully! You can now login.'))
        
        return render_template('confirm_email.html', error='Invalid OTP', email=email)
    
    return render_template('confirm_email.html', email=email)

@app.route('/resend-verification-otp')
def resend_verification_otp():
    if 'verify_email' not in session:
        return redirect(url_for('verify_email'))
    
    email = session.get('verify_email')
    
    # Generate new OTP
    otp = str(random.randint(100000, 999999))
    
    # Update session with new OTP
    session['verify_otp'] = otp
    session['verify_otp_expiry'] = (datetime.now() + timedelta(minutes=10)).timestamp()
    
    # Send OTP via email
    if send_otp_email(email, otp, is_registration=True):
        return redirect(url_for('confirm_email', message='New verification code sent!'))
    else:
        return render_template('confirm_email.html', 
                              error='Failed to send new verification code', 
                              email=email)
# Add these routes to the end of your file, before if __name__ == '__main__':

@app.route('/forgot-password', methods=['GET', 'POST'])
def forgot_password():
    if request.method == 'POST':
        email = request.form.get('email')
        
        # Check if user exists
        user = mongo.db.users.find_one({'email': email})
        
        if not user:
            return render_template('forgot_password.html', error='No account found with this email address')
        
        # Generate OTP
        otp = str(random.randint(100000, 999999))
        
        # Store password reset data in session
        session['reset_email'] = email
        session['reset_otp'] = otp
        session['reset_otp_expiry'] = (datetime.now() + timedelta(minutes=10)).timestamp()
        
        # Send OTP via email
        if send_otp_email(email, otp, is_reset=True):
            return redirect(url_for('reset_password'))
        else:
            return render_template('forgot_password.html', error='Failed to send reset code. Please try again.')
    
    return render_template('forgot_password.html')

@app.route('/reset-password', methods=['GET', 'POST'])
def reset_password():
    if 'reset_email' not in session or 'reset_otp' not in session:
        return redirect(url_for('forgot_password'))
    
    email = session.get('reset_email')
    message = request.args.get('message', '')
    
    if request.method == 'POST':
        user_otp = request.form.get('otp')
        new_password = request.form.get('new_password')
        confirm_password = request.form.get('confirm_password')
        
        if datetime.now().timestamp() > session.get('reset_otp_expiry', 0):
            # Clear session data
            for key in ['reset_email', 'reset_otp', 'reset_otp_expiry']:
                session.pop(key, None)
            return render_template('forgot_password.html', error='Reset code expired. Please request a new one.')
        
        if user_otp != session['reset_otp']:
            return render_template('reset_password.html', email=email, error='Invalid reset code')
        
        if new_password != confirm_password:
            return render_template('reset_password.html', email=email, error='Passwords do not match')
        
        # Validate password
        if len(new_password) < 8:
            return render_template('reset_password.html', email=email, error='Password must be at least 8 characters long')
        
        if not any(char.isupper() for char in new_password):
            return render_template('reset_password.html', email=email, error='Password must contain at least one uppercase letter')
            
        if not any(char.isdigit() for char in new_password):
            return render_template('reset_password.html', email=email, error='Password must contain at least one number')
            
        if not any(char in '!@#$%^&*()-_=+[]{}|;:,.<>?/~`' for char in new_password):
            return render_template('reset_password.html', email=email, error='Password must contain at least one special character')
        
        # Update user's password
        hashed_password = generate_password_hash(new_password)
        mongo.db.users.update_one(
            {'email': email}, 
            {'$set': {'password': hashed_password}}
        )
        
        # Clear reset session data
        for key in ['reset_email', 'reset_otp', 'reset_otp_expiry']:
            session.pop(key, None)
        
        return redirect(url_for('login', message='Password reset successful! Please login with your new password.'))
    
    return render_template('reset_password.html', email=email, message=message)

@app.route('/resend-reset-otp')
def resend_reset_otp():
    if 'reset_email' not in session:
        return redirect(url_for('forgot_password'))
    
    # Generate new OTP
    otp = str(random.randint(100000, 999999))
    
    # Update session with new OTP
    session['reset_otp'] = otp
    session['reset_otp_expiry'] = (datetime.now() + timedelta(minutes=10)).timestamp()
    
    # Send OTP via email
    if send_otp_email(session['reset_email'], otp, is_reset=True):
        return redirect(url_for('reset_password', message='New reset code sent!'))
    else:
        return render_template('reset_password.html', 
                             error='Failed to send new reset code', 
                             email=session.get('reset_email'))
if __name__ == '__main__':
    app.run(debug=True)
