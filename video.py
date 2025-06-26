import cv2
import numpy as np
from ultralytics import YOLO
from PIL import ImageFont, ImageDraw, Image

emotions = [
    "Happiness",
    "Joy",
    "Euphoria",
    "Excitement",
    "Enthusiasm",
    "Contentment",
    "Satisfied",
    "Delighted",
    "Cheerful",
    "Optimistic",
    "Sadness",
    "Sorrow",
    "Grief",
    "Depression",
    "Anxiety",
    "Fear",
    "Anger",
    "Irritation",
    "Frustration",
    "Resentment",
    "Nostalgia",
    "Regret",
    "Guilt",
    "Shame",
    "Embarrassment",
    "Excitement",
    "Annoyance",
    "Happiness",
    "Euphoria",
    "Fear",
    "Apprehension",
    "Stress",
    "Exhaustion",
    "Loneliness",
    "Confusion",
    "Uncertainty",
    "Respect",
    "Reverence",
    "Sympathy",
    "Compassion",
    "Playfulness",
    "Adulthood",
    "Curiosity",
    "Wisdom",
    "Innocence",
    "Experience",
    "Compassion",
    "Empathy",
    "Forgiveness",
    "Pardon",
    "Gratitude",
    "Apathy",
    "Mania",
    "Hypomania",
    "Depression",
    "Trauma",
    "Resilience",
    "Self-discovery",
    "Transformation",
    "Adaptation",
    "Enlightenment",
    "Spiritual awakening"
]

def load_model(weights_path):
    """
    Load YOLO model with specified weights
    """
    try:
        model = YOLO(weights_path)
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def draw_corners(img, bbox, color, thickness=5, l=30, rt=1):
    """Draw corners on bounding box
    Args:
        img: Image to draw on
        bbox: Bounding box coordinates (x1, y1, x2, y2)
        color: Corner color
        thickness: Line thickness
        l: Corner length
        rt: Corner thickness
    """
    x1, y1, x2, y2 = map(int, bbox)
    
    # Top left
    cv2.line(img, (x1, y1), (x1 + l, y1), color, rt)
    cv2.line(img, (x1, y1), (x1, y1 + l), color, rt)
    
    # Top right
    cv2.line(img, (x2, y1), (x2 - l, y1), color, rt)
    cv2.line(img, (x2, y1), (x2, y1 + l), color, rt)
    
    # Bottom left
    cv2.line(img, (x1, y2), (x1 + l, y2), color, rt)
    cv2.line(img, (x1, y2), (x1, y2 - l), color, rt)
    
    # Bottom right
    cv2.line(img, (x2, y2), (x2 - l, y2), color, rt)
    cv2.line(img, (x2, y2), (x2, y2 - l), color, rt)
    
    return img

def get_font(font_path, font_size=20):
    """Load custom TTF font
    Args:
        font_path: Path to TTF font file
        font_size: Font size to use
    Returns:
        ImageFont object
    """
    try:
        font = ImageFont.truetype(font_path, font_size)
        return font
    except Exception as e:
        print(f"Error loading font: {e}")
        return None

def put_text_with_custom_font(img, text, org, font, color=(255, 255, 255)):
    """Draw text using custom font
    Args:
        img: OpenCV image
        text: Text string to draw
        org: Text origin coordinates (x,y)
        font: ImageFont object
        color: Text color in BGR
    """
    # Convert image to PIL format
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    
    # Draw text
    draw.text(org, text, font=font, fill=color[::-1])  # RGB color
    
    # Convert back to OpenCV format
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

# Load YOLOv8 model
model = load_model("SOFTWARE/Yolo-Weights/yolo11x.pt")
if model is None:
    raise Exception("Failed to load YOLO model")

# Load video
cap = cv2.VideoCapture("SOFTWARE/Videos/crowd.mp4")
emotion_index = 0  # Counter for emotions array

# Get video properties
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Define output video writer
output_path = "output_video.mp4"
out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

# Load font once at startup
font_path = "SOFTWARE/fonts/square.ttf"
custom_font = get_font(font_path, font_size=10)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLOv11 inference on the frame with confidence threshold
    results = model(frame, conf=0.5)[0]  # Add confidence threshold here (0.5 = 50%)
    
    # Draw the detections on the frame
    for box in results.boxes:
        x1, y1, x2, y2 = box.xyxy[0]  # Get box coordinates
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        conf = float(box.conf[0])
        cls = int(box.cls[0])
        
        # If person detected, use emotion from array
        if cls < 100:
            label = emotions[emotion_index]
            # Get image shape and processing times
            img_shape = frame.shape[:2]
            raw_text = f"{cls}: {img_shape[0]}x{img_shape[1]} 1 person, {results.speed['inference']:.1f}ms\nSpeed: {results.speed['preprocess']:.1f}ms pre, {results.speed['inference']:.1f}ms inf"
            emotion_index = (emotion_index + 1) % len(emotions)
        else:
            label = f''
            raw_text = ''
            
        # Draw rectangle and label
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 1)
        
        if custom_font:
            frame = put_text_with_custom_font(frame, label, (x1, y1-40), custom_font)
            frame = put_text_with_custom_font(frame, raw_text, (x1, y1+10), custom_font)
        else:
            # Fallback to default OpenCV font
            cv2.putText(frame, label, (x1, y1-30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            cv2.putText(frame, raw_text, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

        # Then draw the corners on top
        frame = draw_corners(frame, (x1, y1, x2, y2), color=(255, 255, 255), l=20, rt=5)
    # Write frame to output video
    out.write(frame)
    
    # Display the frame
    cv2.imshow('Frame', frame)
    
    # Break loop with 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release everything
cap.release()
out.release()
cv2.destroyAllWindows()