#!/usr/bin/env python3
"""
Test script for Building Inspection API
"""

from robust_building_inspection_api import RobustBuildingInspectionAPI
import sys
import os

def main():
    if len(sys.argv) != 2:
        print("Usage: python test_api.py <image_path>")
        return
    
    image_path = sys.argv[1]
    
    if not os.path.exists(image_path):
        print(f"File not found: {image_path}")
        return
    
    print("Loading inspection models...")
    try:
        inspector = RobustBuildingInspectionAPI(
            'crack_detection_model.h5',
            'moisture_detection_model.h5'
        )
    except Exception as e:
        print(f"Failed to initialize API: {e}")
        return
    
    print(f"Testing image: {image_path}")
    result = inspector.predict_image(image_path)
    
    if 'error' in result:
        print(f"Error: {result['error']}")
    else:
        crack = result['crack_detection']
        moisture = result['moisture_detection']
        print(f"Crack: {crack['detected']} (confidence: {crack['confidence']:.3f})")
        print(f"Moisture: {moisture['detected']} (confidence: {moisture['confidence']:.3f})")
        print(f"Risk Level: {result['overall_assessment']['risk_level']}")

if __name__ == "__main__":
    main()
