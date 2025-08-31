#!/bin/bash

# ============================================================================
# Vision Transformer Classification API - Bash Shell Test Script
# ============================================================================

# Configuration - UPDATE THESE PATHS
BASE_URL="http://localhost:8000"
TEST_IMAGE="data/test/images/0000020_jpg.rf.adb61ea3435546f5f3333c46dc9a2b2d.jpg"
TEST_IMAGE_2="data/test/images/train_wheat_100_jpg.rf.4ad13fa54509c3ec305877549e120f64.jpg"
TEST_IMAGE_3="data/test/images/IMG20211024100708_1800_1200_jpg.rf.7171a1ae5c4e9add89c3c679a2c1fc94.jpg"
OUTPUT_DIR="api_test_results"
TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

echo -e "${CYAN}============================================================================${NC}"
echo -e "${CYAN}Vision Transformer Classification API Test Suite${NC}"
echo -e "${CYAN}============================================================================${NC}"
echo
echo -e "${YELLOW}Base URL: ${BASE_URL}${NC}"
echo -e "${YELLOW}Timestamp: ${TIMESTAMP}${NC}"
echo

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Check if curl is available
if ! command -v curl &> /dev/null; then
    echo -e "${RED}Error: curl is not installed${NC}"
    echo -e "${RED}Please install curl: sudo apt-get install curl (Ubuntu/Debian) or brew install curl (macOS)${NC}"
    exit 1
fi

# Check if test images exist
if [ ! -f "$TEST_IMAGE" ]; then
    echo -e "${RED}Warning: Primary test image not found at $TEST_IMAGE${NC}"
    echo -e "${RED}Please update TEST_IMAGE path in the script${NC}"
    echo
fi

echo -e "${CYAN}Starting API tests...${NC}"
echo

# ============================================================================
# Test 1: Health Check
# ============================================================================
echo -e "${CYAN}1. Testing Health Check...${NC}"
if curl -s -X GET "$BASE_URL/api/v1/health" \
  -H "Content-Type: application/json" \
  -w "HTTP Status: %{http_code}\nResponse Time: %{time_total}s\n" \
  -o "$OUTPUT_DIR/health_check_$TIMESTAMP.json"; then
    echo -e "${GREEN}   ✓ Health check completed${NC}"
    echo -e "${GREEN}   Response saved to: $OUTPUT_DIR/health_check_$TIMESTAMP.json${NC}"
else
    echo -e "${RED}   ✗ Health check failed${NC}"
fi
echo

# ============================================================================
# Test 2: Model Information
# ============================================================================
echo -e "${CYAN}2. Testing Model Information...${NC}"
if curl -s -X GET "$BASE_URL/api/v1/model-info" \
  -H "Content-Type: application/json" \
  -w "HTTP Status: %{http_code}\nResponse Time: %{time_total}s\n" \
  -o "$OUTPUT_DIR/model_info_$TIMESTAMP.json"; then
    echo -e "${GREEN}   ✓ Model info retrieved${NC}"
    echo -e "${GREEN}   Response saved to: $OUTPUT_DIR/model_info_$TIMESTAMP.json${NC}"
else
    echo -e "${RED}   ✗ Model info request failed${NC}"
fi
echo

# ============================================================================
# Test 3: Single Grain Classification (only if image exists)
# ============================================================================
if [ -f "$TEST_IMAGE" ]; then
    echo -e "${CYAN}3. Testing Single Grain Classification...${NC}"
    
    # Basic classification
    echo -e "${YELLOW}   3a. Basic classification...${NC}"
    if curl -s -X POST "$BASE_URL/api/v1/classify-grain" \
      -F "file=@$TEST_IMAGE" \
      -w "HTTP Status: %{http_code}\nResponse Time: %{time_total}s\n" \
      -o "$OUTPUT_DIR/single_basic_$TIMESTAMP.json"; then
        echo -e "${GREEN}      ✓ Basic classification completed${NC}"
        echo -e "${GREEN}      Response saved to: $OUTPUT_DIR/single_basic_$TIMESTAMP.json${NC}"
    else
        echo -e "${RED}      ✗ Basic classification failed${NC}"
    fi
    
    # Full feature extraction
    echo -e "${YELLOW}   3b. Full feature classification...${NC}"
    if curl -s -X POST "$BASE_URL/api/v1/classify-grain?extract_features=true&enhance_image=true&return_probabilities=true" \
      -F "file=@$TEST_IMAGE" \
      -w "HTTP Status: %{http_code}\nResponse Time: %{time_total}s\n" \
      -o "$OUTPUT_DIR/single_full_$TIMESTAMP.json"; then
        echo -e "${GREEN}      ✓ Full classification completed${NC}"
        echo -e "${GREEN}      Response saved to: $OUTPUT_DIR/single_full_$TIMESTAMP.json${NC}"
    else
        echo -e "${RED}      ✗ Full classification failed${NC}"
    fi
    
    # Minimal processing
    echo -e "${YELLOW}   3c. Minimal processing...${NC}"
    if curl -s -X POST "$BASE_URL/api/v1/classify-grain?extract_features=false&enhance_image=false&return_probabilities=false" \
      -F "file=@$TEST_IMAGE" \
      -w "HTTP Status: %{http_code}\nResponse Time: %{time_total}s\n" \
      -o "$OUTPUT_DIR/single_minimal_$TIMESTAMP.json"; then
        echo -e "${GREEN}      ✓ Minimal classification completed${NC}"
        echo -e "${GREEN}      Response saved to: $OUTPUT_DIR/single_minimal_$TIMESTAMP.json${NC}"
    else
        echo -e "${RED}      ✗ Minimal classification failed${NC}"
    fi
else
    echo -e "${YELLOW}3. Skipping Single Grain Classification - Test image not found${NC}"
fi
echo

# ============================================================================
# Test 4: Feature Analysis (only if image exists)
# ============================================================================
if [ -f "$TEST_IMAGE" ]; then
    echo -e "${CYAN}4. Testing Feature Analysis...${NC}"
    
    # Both features
    echo -e "${YELLOW}   4a. Both ViT and traditional features...${NC}"
    if curl -s -X POST "$BASE_URL/api/v1/analyze-features?feature_type=both" \
      -F "file=@$TEST_IMAGE" \
      -w "HTTP Status: %{http_code}\nResponse Time: %{time_total}s\n" \
      -o "$OUTPUT_DIR/features_both_$TIMESTAMP.json"; then
        echo -e "${GREEN}      ✓ Both features analysis completed${NC}"
        echo -e "${GREEN}      Response saved to: $OUTPUT_DIR/features_both_$TIMESTAMP.json${NC}"
    else
        echo -e "${RED}      ✗ Both features analysis failed${NC}"
    fi
    
    # ViT features only
    echo -e "${YELLOW}   4b. ViT features only...${NC}"
    if curl -s -X POST "$BASE_URL/api/v1/analyze-features?feature_type=vit" \
      -F "file=@$TEST_IMAGE" \
      -w "HTTP Status: %{http_code}\nResponse Time: %{time_total}s\n" \
      -o "$OUTPUT_DIR/features_vit_$TIMESTAMP.json"; then
        echo -e "${GREEN}      ✓ ViT features analysis completed${NC}"
        echo -e "${GREEN}      Response saved to: $OUTPUT_DIR/features_vit_$TIMESTAMP.json${NC}"
    else
        echo -e "${RED}      ✗ ViT features analysis failed${NC}"
    fi
    
    # Traditional features only
    echo -e "${YELLOW}   4c. Traditional features only...${NC}"
    if curl -s -X POST "$BASE_URL/api/v1/analyze-features?feature_type=traditional" \
      -F "file=@$TEST_IMAGE" \
      -w "HTTP Status: %{http_code}\nResponse Time: %{time_total}s\n" \
      -o "$OUTPUT_DIR/features_traditional_$TIMESTAMP.json"; then
        echo -e "${GREEN}      ✓ Traditional features analysis completed${NC}"
        echo -e "${GREEN}      Response saved to: $OUTPUT_DIR/features_traditional_$TIMESTAMP.json${NC}"
    else
        echo -e "${RED}      ✗ Traditional features analysis failed${NC}"
    fi
else
    echo -e "${YELLOW}4. Skipping Feature Analysis - Test image not found${NC}"
fi
echo

# ============================================================================
# Test 5: Batch Classification (only if multiple images exist)
# ============================================================================
if [ -f "$TEST_IMAGE" ] && [ -f "$TEST_IMAGE_2" ] && [ -f "$TEST_IMAGE_3" ]; then
    echo -e "${CYAN}5. Testing Batch Classification...${NC}"
    
    # Full batch processing
    echo -e "${YELLOW}   5a. Full batch processing...${NC}"
    if curl -s -X POST "$BASE_URL/api/v1/classify-grains-batch?extract_features=true&enhance_images=true&aggregate_results=true" \
      -F "files=@$TEST_IMAGE" \
      -F "files=@$TEST_IMAGE_2" \
      -F "files=@$TEST_IMAGE_3" \
      -w "HTTP Status: %{http_code}\nResponse Time: %{time_total}s\n" \
      -o "$OUTPUT_DIR/batch_full_$TIMESTAMP.json"; then
        echo -e "${GREEN}      ✓ Full batch processing completed${NC}"
        echo -e "${GREEN}      Response saved to: $OUTPUT_DIR/batch_full_$TIMESTAMP.json${NC}"
    else
        echo -e "${RED}      ✗ Full batch processing failed${NC}"
    fi
    
    # Minimal batch processing
    echo -e "${YELLOW}   5b. Minimal batch processing...${NC}"
    if curl -s -X POST "$BASE_URL/api/v1/classify-grains-batch?extract_features=false&enhance_images=false&aggregate_results=false" \
      -F "files=@$TEST_IMAGE" \
      -F "files=@$TEST_IMAGE_2" \
      -w "HTTP Status: %{http_code}\nResponse Time: %{time_total}s\n" \
      -o "$OUTPUT_DIR/batch_minimal_$TIMESTAMP.json"; then
        echo -e "${GREEN}      ✓ Minimal batch processing completed${NC}"
        echo -e "${GREEN}      Response saved to: $OUTPUT_DIR/batch_minimal_$TIMESTAMP.json${NC}"
    else
        echo -e "${RED}      ✗ Minimal batch processing failed${NC}"
    fi
elif [ -f "$TEST_IMAGE" ]; then
    echo -e "${CYAN}5. Testing Single File Batch...${NC}"
    if curl -s -X POST "$BASE_URL/api/v1/classify-grains-batch" \
      -F "files=@$TEST_IMAGE" \
      -w "HTTP Status: %{http_code}\nResponse Time: %{time_total}s\n" \
      -o "$OUTPUT_DIR/batch_single_$TIMESTAMP.json"; then
        echo -e "${GREEN}   ✓ Single file batch completed${NC}"
        echo -e "${GREEN}   Response saved to: $OUTPUT_DIR/batch_single_$TIMESTAMP.json${NC}"
    else
        echo -e "${RED}   ✗ Single file batch failed${NC}"
    fi
else
    echo -e "${YELLOW}5. Skipping Batch Classification - Test images not found${NC}"
fi
echo



# ============================================================================
# Generate Test Summary
# ============================================================================
echo -e "${CYAN}============================================================================${NC}"
echo -e "${CYAN}Test Summary${NC}"
echo -e "${CYAN}============================================================================${NC}"
echo -e "${GREEN}All test results saved to: $OUTPUT_DIR/${NC}"
echo
echo -e "${YELLOW}Files generated:${NC}"
ls -la "$OUTPUT_DIR"/*_"$TIMESTAMP".*
echo

# Create summary file
{
    echo "Vision Transformer Classification API Test Summary"
    echo "Test Date: $(date)"
    echo "Base URL: $BASE_URL"
    echo "Test Image 1: $TEST_IMAGE"
    echo "Test Image 2: $TEST_IMAGE_2"
    echo "Test Image 3: $TEST_IMAGE_3"
    echo
    echo "Files Generated:"
    ls -1 "$OUTPUT_DIR"/*_"$TIMESTAMP".*
} > "$OUTPUT_DIR/test_summary_$TIMESTAMP.txt"

echo -e "${GREEN}Test summary saved to: $OUTPUT_DIR/test_summary_$TIMESTAMP.txt${NC}"
echo
echo -e "${CYAN}Testing completed! Check the output files for detailed results.${NC}"
echo

# Optional: View results
read -p "View results folder? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    if command -v xdg-open &> /dev/null; then
        xdg-open "$OUTPUT_DIR"
    elif command -v open &> /dev/null; then
        open "$OUTPUT_DIR"
    else
        echo -e "${YELLOW}Please navigate to $OUTPUT_DIR to view the results${NC}"
    fi
fi

# Function to parse and display JSON results
display_results() {
    echo -e "${CYAN}Quick Results Summary:${NC}"
    
    # Health check
    if [ -f "$OUTPUT_DIR/health_check_$TIMESTAMP.json" ]; then
        if command -v jq &> /dev/null; then
            echo -e "${YELLOW}Health Status:${NC} $(jq -r '.status // "unknown"' "$OUTPUT_DIR/health_check_$TIMESTAMP.json")"
            echo -e "${YELLOW}Model Loaded:${NC} $(jq -r '.model_loaded // "unknown"' "$OUTPUT_DIR/health_check_$TIMESTAMP.json")"
        fi
    fi
    
    # Single classification result
    if [ -f "$OUTPUT_DIR/single_basic_$TIMESTAMP.json" ]; then
        if command -v jq &> /dev/null; then
            echo -e "${YELLOW}Classification:${NC} $(jq -r '.classification // "unknown"' "$OUTPUT_DIR/single_basic_$TIMESTAMP.json")"
            echo -e "${YELLOW}Confidence:${NC} $(jq -r '.confidence // "unknown"' "$OUTPUT_DIR/single_basic_$TIMESTAMP.json")"
        fi
    fi
}

# Display results if jq is available
if command -v jq &> /dev/null; then
    echo
    display_results
else
    echo -e "${YELLOW}Install 'jq' for JSON parsing and quick results summary${NC}"
fi