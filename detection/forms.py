import os
from django import forms
from django.core.exceptions import ValidationError
from django.contrib.auth.models import User
from django.contrib.auth.forms import UserCreationForm, AuthenticationForm
from .models import VideoRecord

ALLOWED_EXTENSIONS = ['.mp4', '.avi', '.mov', '.mkv']
MAX_FILE_SIZE = 100 * 1024 * 1024  # 100 MB


class VideoUploadForm(forms.ModelForm):
    title = forms.CharField(
        max_length=255,
        required=False,
        widget=forms.TextInput(attrs={
            'class': 'form-control',
            'placeholder': 'Enter video title or road location (optional)',
        })
    )
    original_video = forms.FileField(
        required=True,
        widget=forms.FileInput(attrs={
            'class': 'form-control-file',
            'accept': 'video/mp4,video/x-msvideo,video/quicktime,video/x-matroska',
            'id': 'videoFileInput',
        })
    )

    class Meta:
        model = VideoRecord
        fields = ['title', 'original_video']

    def clean_original_video(self):
        video = self.cleaned_data.get('original_video')
        if not video:
            raise ValidationError("Please select a video file to upload.")

        ext = os.path.splitext(video.name)[1].lower()
        if ext not in ALLOWED_EXTENSIONS:
            raise ValidationError(
                f"Unsupported file format '{ext}'. Supported formats are: {', '.join(ALLOWED_EXTENSIONS)}"
            )

        if video.size > MAX_FILE_SIZE:
            raise ValidationError(
                f"File size exceeds limit of 100MB. Your file is {round(video.size / (1024 * 1024), 1)}MB."
            )

        return video


class UserRegistrationForm(UserCreationForm):
    email = forms.EmailField(
        required=True,
        widget=forms.EmailInput(attrs={'class': 'form-control', 'placeholder': 'Email address'})
    )
    first_name = forms.CharField(
        required=False,
        widget=forms.TextInput(attrs={'class': 'form-control', 'placeholder': 'First name'})
    )
    last_name = forms.CharField(
        required=False,
        widget=forms.TextInput(attrs={'class': 'form-control', 'placeholder': 'Last name'})
    )

    class Meta:
        model = User
        fields = ['username', 'first_name', 'last_name', 'email']

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for field_name, field in self.fields.items():
            field.widget.attrs['class'] = 'form-control'
            field.widget.attrs['placeholder'] = field.label
