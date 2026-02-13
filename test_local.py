cd /Users/aerobe/Documents/SC2079/Main
cat > test_local.py << 'EOF'
#!/usr/bin/env python3
import requests

# Test with sample image
test_image = "../Sample Pic/1758859514_2_C.jpg"

print("Testing API with local image...")
try:
    with open(test_image, 'rb') as f:
        response = requests.post(
            "http://localhost:5000/image",
            files={"file": ("test.jpg", f)},
            timeout=30
        )
    
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.json()}")
    
except Exception as e:
    print(f"Error: {e}")
EOF

python3 test_local.py
