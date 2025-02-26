from flask import Flask
from flask_migrate import Migrate
from flask_sqlalchemy import SQLAlchemy
from flask_wtf.csrf import CSRFProtect

from apps.config import config
from apps.auth import views as auth_views


db = SQLAlchemy()
csrf = CSRFProtect()


def create_app(config_key):
    app = Flask(__name__)
    app.config.from_object(config[config_key])
    db.init_app(app)
    Migrate(app, db)
    app.register_blueprint(auth_views.auth, url_prefix="/auth")
    return app
