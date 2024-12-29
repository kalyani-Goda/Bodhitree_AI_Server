from django.test import TestCase

from .models import Submission

class SubmissionTestCase(TestCase):
    def setUp(self):
        # Setup test data
        Submission.objects.create(id = 2, description = "Test")

    def test_model_field(self):
        # Test your model logic
        obj = Submission.objects.get(id = 2)
        self.assertEqual(obj.description, 'Test')
