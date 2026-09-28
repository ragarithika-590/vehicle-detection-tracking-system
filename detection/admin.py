from django.contrib import admin
from .models import VideoRecord


@admin.register(VideoRecord)
class VideoRecordAdmin(admin.ModelAdmin):
    list_display = ('title', 'filename', 'user', 'status', 'overall_count', 'car_count', 'bus_count', 'truck_count', 'motorcycle_count', 'created_at')
    list_filter = ('status', 'created_at', 'user')
    search_fields = ('title', 'filename', 'user__username', 'error_message')
    readonly_fields = ('created_at', 'updated_at', 'processing_time_seconds', 'total_frames', 'fps')
    ordering = ('-created_at',)
