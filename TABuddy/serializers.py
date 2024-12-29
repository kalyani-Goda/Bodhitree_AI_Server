from rest_framework import serializers

class FileUploadSerializer(serializers.Serializer):
    files = serializers.ListField(
        child=serializers.FileField(),
        write_only=True
    )
    problem_statement = serializers.CharField()
    criteria = serializers.JSONField()