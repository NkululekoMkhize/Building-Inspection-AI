# Building Inspection AI API (Production Ready)

## Features
- Dual Model Analysis: Crack + Moisture detection
- Video Support: Automatic frame extraction & analysis
- Robust Error Handling: Handles corrupt files, large files, format issues
- Multiple Input Types: File paths, bytes, base64, PIL Images

## Models Included
- crack_detection_model.h5 - 99.7% accurate crack detection
- moisture_detection_model.h5 - 89.8% accurate moisture detection

## Quick Start

### Image Analysis
```python
from robust_building_inspection_api import RobustBuildingInspectionAPI

# Initialize
inspector = RobustBuildingInspectionAPI(
    crack_model_path='crack_detection_model.h5',
    moisture_model_path='moisture_detection_model.h5'
)

# Analyze image
result = inspector.predict_image('building.jpg')
```

## Response Format
```json
{
  "crack_detection": {"detected": true, "confidence": 0.995, "severity": "high"},
  "moisture_detection": {"detected": false, "confidence": 0.234, "severity": "none"},
  "overall_assessment": {"condition_score": 0.005, "risk_level": "critical", "urgent_action_required": true},
  "metadata": {"timestamp": 1234567890, "model_version": "production_v2.0", "input_type": "image"}
}
```
