import uuid
from pathlib import Path

from flask import (
    Blueprint,
    current_app,
    redirect,
    render_template,
    send_from_directory,
    url_for,
)
from flask_login import current_user, login_required

from apps.app import db
from apps.crud.models import User
from apps.detector.forms import UploadImageForm
from apps.detector.models import UserImage


dt = Blueprint("detector", __name__, template_folder="templates")


@dt.route("/")
def index():
    user_images = (
        db.session.query(User, UserImage)
        .join(UserImage)
        .filter(User.id == UserImage.user_id)
        .all()
    )
    return render_template("detector/index.html", user_images=user_images)


@dt.route("/images/<path:filename>")
def image_file(filename):
    return send_from_directory(current_app.config["UPLOAD_FOLDER"], filename)


@dt.route("/upload", methods=["GET", "POST"])
@login_required
def upload_image():
    form = UploadImageForm()
    if form.validate_on_submit():
        file = form.image.data
        ext = Path(file.filename).suffix
        image_uuid_file_name = str(uuid.uuid4()) + ext
        image_path = Path(
            current_app.config["UPLOAD_FOLDER"], image_uuid_file_name
        )
        file.save(image_path)
        user_image = UserImage(
            user_id=current_user.id, image_path=image_uuid_file_name
        )
        db.session.add(user_image)
        db.session.commit()
        return redirect(url_for("detector.index"))
    return render_template("detector/upload.html", form=form)
