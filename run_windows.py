import os
from app import app

# Check if certificates exist
cert_path = os.path.join(os.path.dirname(__file__), 'cert.pem')
key_path = os.path.join(os.path.dirname(__file__), 'key.pem')

# If certificates don't exist, try to create them using werkzeug
if not (os.path.exists(cert_path) and os.path.exists(key_path)):
    try:
        from werkzeug.serving import make_ssl_devcert
        make_ssl_devcert('./ssl', host='localhost')
        # Rename the files to match what we expect
        if os.path.exists('./ssl.crt') and os.path.exists('./ssl.key'):
            os.rename('./ssl.crt', cert_path)
            os.rename('./ssl.key', key_path)
            print(f"Generated SSL certificates at {cert_path} and {key_path}")
        else:
            print("Certificate files not created at expected location")
    except Exception as e:
        print(f"Error generating certificates: {e}")
        print("Will run without HTTPS")

# Run the app with or without SSL depending on certificate availability
if os.path.exists(cert_path) and os.path.exists(key_path):
    print("Starting Flask with HTTPS support...")
    app.run(debug=True, host='0.0.0.0', port=5000, ssl_context=(cert_path, key_path))
else:
    print("Starting Flask without HTTPS support...")
    app.run(debug=True, host='0.0.0.0', port=5000)