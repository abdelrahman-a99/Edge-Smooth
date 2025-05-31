from flask import Flask
from config import Config

def create_app(config_class=Config):
    app = Flask(__name__)
    app.config.from_object(config_class)

    # Ensure upload directories exist
    app.config['UPLOAD_FOLDER'] = 'app/static/uploads'
    app.config['PROCESSED_FOLDER'] = 'app/static/processed'

    # Import and register the app
    from app.app import app as main_app
    app = main_app

    return app 