import os
from django.db import models
from django.contrib.auth.models import User


class VideoRecord(models.Model):
    STATUS_CHOICES = [
        ('PENDING', 'Pending'),
        ('PROCESSING', 'Processing'),
        ('COMPLETED', 'Completed'),
        ('FAILED', 'Failed'),
    ]

    user = models.ForeignKey(User, on_delete=models.SET_NULL, null=True, blank=True, related_name='video_records')
    title = models.CharField(max_length=255, default='Traffic Analysis Video')
    original_video = models.FileField(upload_to='uploads/')
    processed_video = models.FileField(upload_to='processed/', blank=True, null=True)
    filename = models.CharField(max_length=255, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    # Counts by category
    car_count = models.IntegerField(default=0)
    bus_count = models.IntegerField(default=0)
    truck_count = models.IntegerField(default=0)
    motorcycle_count = models.IntegerField(default=0)
    overall_count = models.IntegerField(default=0)
    
    # Video details & processing metadata
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='PENDING')
    error_message = models.TextField(blank=True, null=True)
    fps = models.FloatField(default=0.0)
    total_frames = models.IntegerField(default=0)
    current_frame = models.IntegerField(default=0)
    progress_percent = models.IntegerField(default=0)
    processing_time_seconds = models.FloatField(default=0.0)

    class Meta:
        ordering = ['-created_at']
        verbose_name = 'Video Record'
        verbose_name_plural = 'Video Records'

    def __str__(self):
        return f"{self.title} - {self.filename} ({self.get_status_display()})"

    def save(self, *args, **kwargs):
        if self.original_video and not self.filename:
            self.filename = os.path.basename(self.original_video.name)
        self.overall_count = self.car_count + self.bus_count + self.truck_count + self.motorcycle_count
        super().save(*args, **kwargs)
