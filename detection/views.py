import os
import time
import threading
import logging
from django.shortcuts import render, redirect, get_object_or_404
from django.contrib import messages
from django.contrib.auth import login, logout, authenticate
from django.contrib.auth.forms import AuthenticationForm
from django.contrib.auth.decorators import login_required
from django.http import StreamingHttpResponse, JsonResponse, HttpResponse, Http404
from django.urls import reverse
from django.db import close_old_connections
from django.db.models import Sum

from .models import VideoRecord
from .forms import VideoUploadForm, UserRegistrationForm
from .services import VideoProcessor, LiveWebcamTracker

logger = logging.getLogger(__name__)

# Global tracker instance for live webcam feed
_live_tracker_instance = None

def get_live_tracker():
    global _live_tracker_instance
    if _live_tracker_instance is None:
        _live_tracker_instance = LiveWebcamTracker(camera_id=0)
    return _live_tracker_instance


def db_retry_call(fn, max_retries=5, delay=0.3):
    """
    Executes a database operation with retries to safely handle
    SQLite table lock contention in multi-threaded environments.
    """
    for attempt in range(max_retries):
        try:
            close_old_connections()
            return fn()
        except Exception as e:
            if 'locked' in str(e).lower() and attempt < max_retries - 1:
                time.sleep(delay * (attempt + 1))
            else:
                raise e


def _run_background_video_processing(record_pk):
    """
    Non-blocking background thread worker for YOLOv8 + ByteTrack processing.
    Executes in a separate thread so Django HTTP requests return immediately.
    """
    try:
        def init_record():
            rec = VideoRecord.objects.get(pk=record_pk)
            if rec.status in ['COMPLETED', 'FAILED']:
                return None
            rec.status = 'PROCESSING'
            rec.progress_percent = 0
            rec.save(update_fields=['status', 'progress_percent', 'updated_at'])
            return rec

        record = db_retry_call(init_record)
        if record is None:
            return

        logger.info(f"[JOB #{record_pk}] Starting background processing for: {record.filename}")
        processor = VideoProcessor(input_path=record.original_video.path)
        last_saved_pct = -1

        def progress_callback(current_frame, total_frames, pct, counts):
            nonlocal last_saved_pct
            if pct - last_saved_pct >= 2 or pct == 100:
                last_saved_pct = pct
                try:
                    def update_progress():
                        VideoRecord.objects.filter(pk=record_pk).update(
                            progress_percent=pct,
                            current_frame=current_frame,
                            total_frames=total_frames,
                            car_count=counts.get('car', 0),
                            bus_count=counts.get('bus', 0),
                            truck_count=counts.get('truck', 0),
                            motorcycle_count=counts.get('motorcycle', 0),
                            overall_count=sum(counts.values())
                        )
                    db_retry_call(update_progress, max_retries=3, delay=0.2)
                except Exception as db_err:
                    logger.warning(f"[JOB #{record_pk}] DB progress update warning: {db_err}")

        results = processor.process_video(progress_callback=progress_callback)

        def save_final():
            final_record = VideoRecord.objects.get(pk=record_pk)
            final_record.processed_video.name = results['relative_output_path']
            final_record.car_count = results['car_count']
            final_record.bus_count = results['bus_count']
            final_record.truck_count = results['truck_count']
            final_record.motorcycle_count = results['motorcycle_count']
            final_record.overall_count = results['overall_count']
            final_record.fps = results['fps']
            final_record.total_frames = results['total_frames']
            final_record.current_frame = results['total_frames']
            final_record.progress_percent = 100
            final_record.processing_time_seconds = results['processing_time_seconds']
            final_record.status = 'COMPLETED'
            final_record.error_message = None
            final_record.save()

        db_retry_call(save_final)
        logger.info(f"[JOB #{record_pk}] Completed in {results['processing_time_seconds']}s ({results['overall_count']} vehicles detected)")

    except VideoRecord.DoesNotExist:
        logger.info(f"[JOB #{record_pk}] Video record no longer exists. Thread exiting.")
    except Exception as e:
        logger.error(f"[JOB #{record_pk}] Video processing failed: {e}", exc_info=True)
        try:
            def save_failure():
                VideoRecord.objects.filter(pk=record_pk).update(
                    status='FAILED',
                    error_message=str(e)
                )
            db_retry_call(save_failure)
        except Exception as db_fail_err:
            logger.error(f"[JOB #{record_pk}] Failed to save FAILED status: {db_fail_err}")
    finally:
        close_old_connections()


# ============================================================================
# PUBLIC VIEWS
# ============================================================================

def home_view(request):
    """
    Public Home Page.
    """
    recent_videos = VideoRecord.objects.filter(status='COMPLETED').order_by('-created_at')[:3]
    total_processed = VideoRecord.objects.filter(status='COMPLETED').count()
    overall_vehicles = VideoRecord.objects.filter(status='COMPLETED').aggregate(Sum('overall_count'))['overall_count__sum'] or 0

    context = {
        'recent_videos': recent_videos,
        'total_processed': total_processed,
        'overall_vehicles': overall_vehicles,
    }
    return render(request, 'detection/home.html', context)


def about_view(request):
    """
    Public About Page detailing technical architecture & specifications.
    """
    return render(request, 'detection/about.html')


def register_view(request):
    """
    Public User Registration View. Redirects authenticated users to dashboard.
    """
    if request.user.is_authenticated:
        return redirect('dashboard')

    if request.method == 'POST':
        form = UserRegistrationForm(request.POST)
        if form.is_valid():
            user = form.save()
            login(request, user)
            messages.success(request, f"Welcome to Vehicle Track AI, {user.username}! Account created successfully.")
            return redirect('dashboard')
        else:
            messages.error(request, "Please correct the registration errors below.")
    else:
        form = UserRegistrationForm()

    return render(request, 'detection/register.html', {'form': form})


def login_view(request):
    """
    Public User Login View. Redirects authenticated users to dashboard.
    """
    if request.user.is_authenticated:
        return redirect('dashboard')

    if request.method == 'POST':
        form = AuthenticationForm(request, data=request.POST)
        if form.is_valid():
            username = form.cleaned_data.get('username')
            password = form.cleaned_data.get('password')
            user = authenticate(username=username, password=password)
            if user is not None:
                login(request, user)
                messages.success(request, f"Welcome back, {username}!")
                next_url = request.GET.get('next', 'dashboard')
                return redirect(next_url)
            else:
                messages.error(request, "Invalid username or password.")
        else:
            messages.error(request, "Invalid username or password.")
    else:
        form = AuthenticationForm()

    return render(request, 'detection/login.html', {'form': form})


def logout_view(request):
    """
    User Logout View.
    """
    logout(request)
    messages.info(request, "You have been logged out successfully.")
    return redirect('home')


# ============================================================================
# PROTECTED AUTHENTICATED VIEWS
# ============================================================================

@login_required
def dashboard_view(request):
    """
    Authenticated User Dashboard.
    """
    user_videos = VideoRecord.objects.filter(user=request.user)
    completed_videos = user_videos.filter(status='COMPLETED')

    total_videos = user_videos.count()
    total_cars = completed_videos.aggregate(Sum('car_count'))['car_count__sum'] or 0
    total_buses = completed_videos.aggregate(Sum('bus_count'))['bus_count__sum'] or 0
    total_trucks = completed_videos.aggregate(Sum('truck_count'))['truck_count__sum'] or 0
    total_motorcycles = completed_videos.aggregate(Sum('motorcycle_count'))['motorcycle_count__sum'] or 0
    total_overall = completed_videos.aggregate(Sum('overall_count'))['overall_count__sum'] or 0

    context = {
        'user_videos': user_videos[:5],
        'total_videos': total_videos,
        'total_cars': total_cars,
        'total_buses': total_buses,
        'total_trucks': total_trucks,
        'total_motorcycles': total_motorcycles,
        'total_overall': total_overall,
    }
    return render(request, 'detection/dashboard.html', context)


@login_required
def upload_video_view(request):
    """
    Protected Video Upload View. Saves record and triggers background worker thread.
    """
    if request.method == 'POST':
        form = VideoUploadForm(request.POST, request.FILES)
        if form.is_valid():
            try:
                record = form.save(commit=False)
                record.user = request.user
                
                if not record.title:
                    record.title = os.path.splitext(record.original_video.name)[0].replace('_', ' ').title()

                record.status = 'PENDING'
                record.progress_percent = 0
                record.save()

                # Launch non-blocking background thread
                thread = threading.Thread(
                    target=_run_background_video_processing,
                    args=(record.pk,),
                    daemon=True
                )
                thread.start()

                logger.info(f"Background thread launched for VideoRecord ID: {record.pk}")
                return redirect('processing_page', pk=record.pk)

            except Exception as e:
                logger.error(f"Error saving uploaded video: {e}")
                messages.error(request, f"Failed to save uploaded video: {str(e)}")
        else:
            for field, errors in form.errors.items():
                for error in errors:
                    messages.error(request, f"{error}")
    else:
        form = VideoUploadForm()

    return render(request, 'detection/upload.html', {'form': form})


@login_required
def process_video_view(request, pk):
    """
    Protected trigger view for initiating or re-triggering processing.
    """
    record = get_object_or_404(VideoRecord, pk=pk)

    # Ownership / Admin check
    if record.user and record.user != request.user and not request.user.is_staff:
        messages.error(request, "You do not have permission to process this video record.")
        return redirect('dashboard')

    if record.status == 'COMPLETED':
        return redirect('results', pk=record.pk)

    if record.status in ['PENDING', 'FAILED']:
        record.status = 'PENDING'
        record.save()
        thread = threading.Thread(
            target=_run_background_video_processing,
            args=(record.pk,),
            daemon=True
        )
        thread.start()

    return redirect('processing_page', pk=record.pk)


@login_required
def processing_page_view(request, pk):
    """
    Protected View rendering progress bar and live polling UI.
    """
    record = get_object_or_404(VideoRecord, pk=pk)

    if record.user and record.user != request.user and not request.user.is_staff:
        messages.error(request, "You do not have permission to view this processing job.")
        return redirect('dashboard')

    if record.status == 'COMPLETED':
        return redirect('results', pk=record.pk)

    return render(request, 'detection/processing.html', {'record': record})


@login_required
def processing_status_api(request, pk):
    """
    Protected JSON API endpoint polled by the frontend progress page.
    """
    record = get_object_or_404(VideoRecord, pk=pk)

    if record.user and record.user != request.user and not request.user.is_staff:
        return JsonResponse({'error': 'Unauthorized'}, status=403)

    return JsonResponse({
        'status': record.status,
        'progress': record.progress_percent,
        'current_frame': record.current_frame,
        'total_frames': record.total_frames,
        'car_count': record.car_count,
        'bus_count': record.bus_count,
        'truck_count': record.truck_count,
        'motorcycle_count': record.motorcycle_count,
        'overall_count': record.overall_count,
        'results_url': reverse('results', kwargs={'pk': record.pk}),
        'error_message': record.error_message,
    })


@login_required
def results_view(request, pk):
    """
    Protected Results View displaying detection counts and video player.
    """
    record = get_object_or_404(VideoRecord, pk=pk)

    if record.user and record.user != request.user and not request.user.is_staff:
        messages.error(request, "You do not have permission to view these private video results.")
        return redirect('dashboard')

    context = {
        'record': record,
    }
    return render(request, 'detection/results.html', context)


@login_required
def history_view(request):
    """
    Protected Video History Archive listing previous records.
    """
    if request.user.is_staff:
        videos = VideoRecord.objects.all()
    else:
        videos = VideoRecord.objects.filter(user=request.user)

    query = request.GET.get('q', '')
    if query:
        videos = videos.filter(title__icontains=query)

    context = {
        'videos': videos,
        'query': query,
    }
    return render(request, 'detection/history.html', context)


@login_required
def delete_video_view(request, pk):
    """
    Protected View to safely delete a video record and media files.
    """
    record = get_object_or_404(VideoRecord, pk=pk)

    if record.user and record.user != request.user and not request.user.is_staff:
        messages.error(request, "You do not have permission to delete this video record.")
        return redirect('history')

    if request.method == 'POST':
        if record.original_video and os.path.exists(record.original_video.path):
            try:
                os.remove(record.original_video.path)
            except Exception as e:
                logger.warning(f"Failed to delete original video: {e}")
                
        if record.processed_video and os.path.exists(record.processed_video.path):
            try:
                os.remove(record.processed_video.path)
            except Exception as e:
                logger.warning(f"Failed to delete processed video: {e}")

        record.delete()
        messages.success(request, "Video record deleted successfully.")
        return redirect('history')

    return render(request, 'detection/delete_confirm.html', {'record': record})


@login_required
def live_tracker_view(request):
    """
    Protected Live Webcam Tracking Page.
    """
    return render(request, 'detection/live_tracker.html')


@login_required
def live_stream_feed(request):
    """
    Protected Live Webcam MJPEG stream feed.
    """
    tracker = get_live_tracker()
    return StreamingHttpResponse(
        tracker.generate_frames(),
        content_type='multipart/x-mixed-replace; boundary=frame'
    )


@login_required
def live_reset_counter(request):
    """
    Protected Live Counter Reset API endpoint.
    """
    if request.method == 'POST':
        tracker = get_live_tracker()
        tracker.reset_counts()
        return JsonResponse({'status': 'success', 'message': 'Live tracking counts reset successfully.'})
    return JsonResponse({'status': 'error', 'message': 'Invalid request method.'}, status=400)
