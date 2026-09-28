import os
import tempfile
import cv2
import numpy as np
from django.test import TransactionTestCase, Client
from django.urls import reverse
from django.core.files.uploadedfile import SimpleUploadedFile
from django.contrib.auth.models import User
from detection.models import VideoRecord
from detection.forms import VideoUploadForm
from detection.services.video_processor import VideoProcessor


def create_dummy_mp4_video(file_path, num_frames=15, width=320, height=240, fps=15):
    """
    Utility function to create a small valid MP4 video file for testing.
    """
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(file_path, fourcc, fps, (width, height))
    for i in range(num_frames):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        cv2.rectangle(frame, (50 + i * 2, 50), (150 + i * 2, 150), (255, 120, 0), -1)
        cv2.putText(frame, f"Test Frame {i}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        out.write(frame)
    out.release()


class VehicleDetectionSystemTestCase(TransactionTestCase):

    def setUp(self):
        self.client = Client()
        self.user = User.objects.create_user(
            username='teststudent',
            email='student@example.com',
            password='Password123!'
        )

    def test_public_vs_protected_route_authorization(self):
        """
        Verify that public URLs are accessible by guests, but protected URLs
        strictly redirect unauthenticated users to the login page.
        """
        # Public URLs -> Must return 200 OK
        public_urls = [
            reverse('home'),
            reverse('about'),
            reverse('login'),
            reverse('register'),
        ]
        for url in public_urls:
            response = self.client.get(url)
            self.assertEqual(response.status_code, 200, f"Public URL {url} failed with {response.status_code}")

        # Protected URLs for guests -> Must redirect to login (HTTP 302)
        protected_urls = [
            reverse('dashboard'),
            reverse('upload'),
            reverse('live_tracker'),
            reverse('history'),
        ]
        for url in protected_urls:
            response = self.client.get(url)
            self.assertEqual(response.status_code, 302, f"Protected URL {url} allowed unauthenticated access!")
            self.assertIn('/login/', response.url)

    def test_authenticated_user_route_access(self):
        """
        Verify logged-in user can access protected features.
        """
        self.client.login(username='teststudent', password='Password123!')
        protected_urls = [
            reverse('dashboard'),
            reverse('upload'),
            reverse('live_tracker'),
            reverse('history'),
        ]
        for url in protected_urls:
            response = self.client.get(url)
            self.assertEqual(response.status_code, 200, f"Authenticated URL {url} failed with {response.status_code}")

    def test_tc03_empty_upload_validation(self):
        """
        TC03: Submit video upload form without selecting a file.
        """
        self.client.login(username='teststudent', password='Password123!')
        response = self.client.post(reverse('upload'), {})
        self.assertEqual(response.status_code, 200)
        form = response.context['form']
        self.assertFalse(form.is_valid())
        self.assertIn('original_video', form.errors)

    def test_tc02_invalid_file_format_rejection(self):
        """
        TC02: Upload invalid file format.
        """
        invalid_file = SimpleUploadedFile(
            "malicious_script.exe",
            b"echo Hello World",
            content_type="application/octet-stream"
        )
        form = VideoUploadForm(data={'title': 'Invalid File'}, files={'original_video': invalid_file})
        self.assertFalse(form.is_valid())
        self.assertIn('original_video', form.errors)
        self.assertIn('Unsupported file format', form.errors['original_video'][0])

    def test_non_blocking_video_upload_and_status_api(self):
        """
        TC01, TC09 & Non-Blocking API: Upload video, immediately redirect to processing page,
        and test JSON polling status API.
        """
        self.client.login(username='teststudent', password='Password123!')

        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp:
            create_dummy_mp4_video(tmp.name, num_frames=10)
            tmp.close()

            with open(tmp.name, 'rb') as video_file:
                uploaded = SimpleUploadedFile(
                    "traffic_sample.mp4",
                    video_file.read(),
                    content_type="video/mp4"
                )
                response = self.client.post(
                    reverse('upload'),
                    {'title': 'Async Traffic Test', 'original_video': uploaded}
                )

            os.remove(tmp.name)

        # Upload view must return HTTP 302 redirect immediately to processing page
        self.assertEqual(response.status_code, 302)
        record = VideoRecord.objects.filter(title='Async Traffic Test').first()
        self.assertIsNotNone(record)

        # Test status API
        status_response = self.client.get(reverse('processing_status_api', kwargs={'pk': record.pk}))
        self.assertEqual(status_response.status_code, 200)
        data = status_response.json()
        self.assertIn('status', data)
        self.assertIn('progress', data)

    def test_video_processing_pipeline(self):
        """
        TC04, TC05, TC06, TC07, TC08: VideoProcessor execution and results rendering.
        """
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp:
            create_dummy_mp4_video(tmp.name, num_frames=10)
            tmp.close()

            processor = VideoProcessor(input_path=tmp.name)
            results = processor.process_video()

            os.remove(tmp.name)

        self.assertTrue(os.path.exists(results['output_path']))

        self.client.login(username='teststudent', password='Password123!')
        record = VideoRecord.objects.create(
            user=self.user,
            title='Pipeline Test Video',
            car_count=results['car_count'],
            bus_count=results['bus_count'],
            truck_count=results['truck_count'],
            motorcycle_count=results['motorcycle_count'],
            overall_count=results['overall_count'],
            processed_video=results['relative_output_path'],
            status='COMPLETED'
        )

        response = self.client.get(reverse('results', kwargs={'pk': record.pk}))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'Pipeline Test Video')

        if os.path.exists(results['output_path']):
            os.remove(results['output_path'])
