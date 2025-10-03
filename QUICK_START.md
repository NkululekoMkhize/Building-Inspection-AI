# 🏢 Building Inspection AI - Quick Start Guide

## Package Contents:
- 📁 MODELS/ - AI models and API
- 📁 TEST_IMAGES/ - Sample images for testing  
- 📁 DOCUMENTATION/ - Setup guides and requirements

## 🚀 Immediate Testing:
```bash
# Install dependencies
pip install -r DOCUMENTATION/requirements.txt

# Test the API
python DOCUMENTATION/test_api.py TEST_IMAGES/clear_wall.jpg
python DOCUMENTATION/test_api.py TEST_IMAGES/cracked_concrete.jpg
```

## 📊 Expected Results:
- ✅ clear_wall.jpg: No defects detected
- ✅ cracked_concrete.jpg: Crack + Moisture detected (CRITICAL)
- ✅ wet_ceiling.jpg: Crack + Moisture detected (CRITICAL) 
- ✅ mixed_defects.jpg: Crack + Moisture detected (CRITICAL)

## 🔧 Integration:
```python
from MODELS.robust_building_inspection_api import RobustBuildingInspectionAPI

inspector = RobustBuildingInspectionAPI(
    crack_model_path='MODELS/crack_detection_model.h5',
    moisture_model_path='MODELS/moisture_detection_model.h5'
)

result = inspector.predict_image('building_photo.jpg')
```

## 📅 Package Generated: 2025-10-03 21:04:02
