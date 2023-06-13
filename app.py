from fastapi import FastAPI
import cv2
import numpy as np
from typing import List, Tuple
import base64
import uvicorn
from pathlib import Path
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

origins = ['*']

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_methods=['*'],
    allow_headers=['*'],
)   

def image_template_matching(template_path, image_path)-> Tuple[Tuple[int, int, int, int], float]:
    # Load the template image, convert it to grayscale, and detect edges
    template = cv2.imread(template_path)
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    template_edges = cv2.Canny(template_gray, 50, 200)
    (tH, tW) = template_gray.shape[:2]

    # Load the image, convert it to grayscale
    image = cv2.imread(image_path)
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect edges in the image
    image_edges = cv2.Canny(image_gray, 50, 200)

    # Perform template matching with the original template
    result = cv2.matchTemplate(image_edges, template_edges, cv2.TM_CCOEFF)
    (_, maxVal, _, maxLoc) = cv2.minMaxLoc(result)

    # Perform template matching with the mirrored template
    template_mirrored = cv2.flip(template_edges, 1)
    result_mirrored = cv2.matchTemplate(image_edges, template_mirrored, cv2.TM_CCOEFF)
    (_, maxVal_mirrored, _, maxLoc_mirrored) = cv2.minMaxLoc(result_mirrored)

    # Determine the best match between the original and mirrored template
    if maxVal > maxVal_mirrored:
        (startX, startY) = (int(maxLoc[0]), int(maxLoc[1]))
        (endX, endY) = (int(maxLoc[0] + tW), int(maxLoc[1] + tH))
    else:
        (startX, startY) = (int(maxLoc_mirrored[0]), int(maxLoc_mirrored[1]))
        (endX, endY) = (int(maxLoc_mirrored[0] + tW), int(maxLoc_mirrored[1] + tH))

    return (startX, startY, endX, endY), maxVal

def detect_pentagon(image)-> bool:
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    threshold_value, thrash = cv2.threshold(img_gray, 240, 255, cv2.CHAIN_APPROX_NONE)
    contours, hierarchy = cv2.findContours(thrash, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    for contour in contours:
        approx = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True), True)
        if len(approx) == 5:
            return True

    return False

def find_best_match(templates, image_path) -> dict:
    max_val = -float("inf")
    best_match_coordinates = None
    energy_arrow_found = False
    image=None

    # Iterate over the templates and find the best match
    for template_path in templates:
        coordinates, match_strength = image_template_matching(template_path, image_path)
        # Load the small portion of the image bounded by the bounding box
        (startX, startY, endX, endY) = coordinates
        small_image = cv2.imread(image_path)[startY:endY, startX:endX]
        # Check if the small portion of the image contains a pentagon
        if detect_pentagon(small_image):
            if match_strength > max_val:
                max_val = match_strength
                best_match_coordinates = coordinates
                energy_arrow_found = True

    # Load the original image
    image = cv2.imread(image_path)

    # Draw the bounding box of the best match on the original image
    if best_match_coordinates is not None:
        (startX, startY, endX, endY) = best_match_coordinates
        cv2.rectangle(image, (startX, startY), (endX, endY), (0, 0, 255), 2)

    # Calculate the center coordinates
    centerX = int((startX + endX) / 2)
    centerY = int((startY + endY) / 2)

    # Create a dictionary with the results
    result = {
        "image": image,
        "matching_strength": max_val,
        "center_coordinates": (centerX, centerY),
        "energy_arrow_found": energy_arrow_found
    }

    return result

@app.get("/ai/single-arrow-match")
def get_best_match() -> dict:
    # Get the directory path of the script file
    script_dir = Path(__file__).resolve().parent

    # List of template paths
    template_dir = script_dir / "assets" / "energy arrows"
    templates = [
        template_dir / "A arrow.png",
        template_dir / "B arrow.png",
        template_dir / "C arrow.png",
        template_dir / "D arrow.png",
        template_dir / "E arrow.png",
        template_dir / "F arrow.png",
        template_dir / "G arrow.png",
    ]

    # Load the main image
    image_path = script_dir / "assets" / "test screenshots" / "D Label.png"

    # Find the best match
    result = find_best_match([str(template) for template in templates], str(image_path))

    # Prepare the response data as a dictionary
    response_data = {
        "matching_strength": result["matching_strength"],
        "center_coordinates": result["center_coordinates"],
        "energy_arrow_found": result["energy_arrow_found"]
    }

    # Encode the image as base64 and include it in the response
    image = result["image"]
    retval, buffer = cv2.imencode('.png', image)
    image_base64 = base64.b64encode(buffer).decode('utf-8')
    response_data["image"] = image_base64

    return response_data
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
