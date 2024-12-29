from django.db import models

class Problem(models.Model):
    id = models.IntegerField(primary_key=True) # Program Activity ID
    problem_statement = models.TextField(default="No problem statement provided.")
    user_id = models.CharField(max_length=255, default="Unknown User ID")
    lab_id = models.IntegerField()
    totalDataPoints = models.IntegerField(default=0)  
    isTrained = models.BooleanField(default=False)  
    def __str__(self):
        return f"Problem {self.id}: {self.problem_statement[:30]}... created by {self.user_id}"   

class Submission(models.Model):
    id = models.AutoField(primary_key=True)
    problem = models.ForeignKey(Problem, on_delete=models.CASCADE)
    student_id = models.CharField(max_length=255, default="Unknown Student ID")
    dummy_id = models.CharField(max_length=255, default="Unknown Dummy ID")
    filename = models.CharField(max_length=255, default="default_filename.cpp")
    source_code = models.TextField(default="No source code provided.")
    def __str__(self):
        # Replace newlines with spaces to make the source code single-line
        single_line_source_code = self.source_code.replace('\n', ' ').replace('\r', '')
        return f"Submission of {self.dummy_id}: {self.filename}: {single_line_source_code}"
    def __repr__(self):
        return f"<Submission(id={self.id}, name='{self.filename}', description='{self.source_code}')>"

class Criteria(models.Model):
    id = models.IntegerField(primary_key=True)
    problem = models.ForeignKey(Problem, on_delete=models.CASCADE)
    title = models.CharField(max_length=255, default="Default Criterion Title")
    description = models.TextField(default="No description provided.")
    def __str__(self):
        return f"{self.title} (ID: {self.id}) (Description: {self.description})"

class Rating(models.Model):
    id = models.IntegerField(primary_key=True)
    title = models.CharField(max_length=255, default="Default Title")
    description = models.TextField(default="No description provided.")
    marks = models.IntegerField(default=0)
    criteria = models.ForeignKey(Criteria, on_delete=models.CASCADE)
    def __str__(self):
        return f"{self.title} (ID: {self.id}) (Description: {self.description}) (Marks: {self.marks})"

class GradingHistory(models.Model):
    id = models.AutoField(primary_key=True)
    submission = models.ForeignKey(Submission, on_delete=models.CASCADE)
    criteria = models.ForeignKey(Criteria, on_delete=models.CASCADE)
    manual_rating = models.ForeignKey(Rating, on_delete=models.CASCADE, related_name='manual_grading_histories')
    ai_rating = models.ForeignKey(Rating, on_delete=models.CASCADE, related_name='ai_grading_histories')
    manual_comments = models.TextField(default="No comments.")
    ai_comments = models.TextField(default="No AI reasonings.")
