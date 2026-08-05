"""
One-time connection test: confirms our API key + project name actually
work before we touch build_features.py. Doesn't write anything.
"""
import os
from dotenv import load_dotenv
import hopsworks

load_dotenv()

project = hopsworks.login(
    api_key_value=os.getenv("HOPSWORKS_API_KEY"),
    project=os.getenv("HOPSWORKS_PROJECT_NAME"),
)

print(f"Connected successfully to project: {project.name}")
fs = project.get_feature_store()
print(f"Feature store retrieved: {fs.name}")
