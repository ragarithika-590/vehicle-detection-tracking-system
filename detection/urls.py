from django.urls import path
from . import views

urlpatterns = [
    # Public Routes
    path('', views.home_view, name='home'),
    path('about/', views.about_view, name='about'),
    path('register/', views.register_view, name='register'),
    path('login/', views.login_view, name='login'),
    path('logout/', views.logout_view, name='logout'),

    # Authenticated Protected Routes
    path('dashboard/', views.dashboard_view, name='dashboard'),
    path('upload/', views.upload_video_view, name='upload'),
    path('process/<int:pk>/', views.process_video_view, name='process_video'),
    path('processing/<int:pk>/', views.processing_page_view, name='processing_page'),
    path('api/status/<int:pk>/', views.processing_status_api, name='processing_status_api'),
    path('results/<int:pk>/', views.results_view, name='results'),
    path('history/', views.history_view, name='history'),
    path('delete/<int:pk>/', views.delete_video_view, name='delete_video'),
    path('live/', views.live_tracker_view, name='live_tracker'),
    path('live/feed/', views.live_stream_feed, name='live_stream_feed'),
    path('live/reset/', views.live_reset_counter, name='live_reset_counter'),
]
