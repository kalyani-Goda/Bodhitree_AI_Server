from django.contrib import admin
from .models import *

# Register your models here.
admin.site.register(Rating)
admin.site.register(Criteria) 
admin.site.register(Problem)  
admin.site.register(Submission)
admin.site.register(GradingHistory)