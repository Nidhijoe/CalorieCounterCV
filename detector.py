import numpy as np
from PIL import Image, ImageDraw
from ultralytics import YOLO
import torch
from torchvision import transforms, models
from torchvision.models import MobileNet_V2_Weights

COLORS = [
    (255, 99, 71), (60, 179, 113), (30, 144, 255),
    (255, 165, 0), (186, 85, 211), (255, 215, 0),
    (0, 206, 209), (255, 105, 180)
]

yolo_model = YOLO("yolov8n.pt")
weights = MobileNet_V2_Weights.DEFAULT
classifier = models.mobilenet_v2(weights=weights)
classifier.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

CONTAINER_IDS = {41, 42, 43, 44, 45, 69, 60}

FOOD_COCO = {
    46: "banana", 47: "apple", 48: "sandwich", 49: "orange",
    50: "broccoli", 51: "carrot", 52: "hot dog", 53: "pizza",
    54: "donut", 55: "cake",
}

IMAGENET_FOOD_MAP = {
    924: "banana", 948: "apple", 963: "orange", 954: "egg",
    923: "rice", 959: "noodles", 926: "pasta", 933: "sandwich",
    934: "hot dog", 935: "burger", 936: "burger", 960: "pizza",
    957: "donut", 956: "cake", 925: "salad", 937: "carrot",
    938: "broccoli", 943: "chicken", 932: "soup", 930: "bread",
    849: "fried rice", 550: "omelette",
}

# These are commonly misdetected by MobileNet on Indian food — suppress them
# unless YOLO also confirms them with high confidence
WESTERN_FALSE_POSITIVES = {"pasta", "salad", "noodles", "sandwich", "burger",
                            "hot dog", "pizza", "soup", "bread", "broccoli",
                            "carrot", "omelette", "fried rice"}


def _analyze_region(image: Image.Image) -> list:
    """Multi-region color analysis tuned for Indian food."""
    img = image.resize((200, 200))
    arr = np.array(img).astype(float)
    h, w = arr.shape[:2]

    regions = {
        "full":         arr,
        "top_left":     arr[:h//2, :w//2],
        "top_right":    arr[:h//2, w//2:],
        "bottom_left":  arr[h//2:, :w//2],
        "bottom_right": arr[h//2:, w//2:],
        "center":       arr[h//4:3*h//4, w//4:3*w//4],
    }

    found = set()

    for _, region in regions.items():
        if region.size == 0:
            continue
        r = region[:, :, 0].mean()
        g = region[:, :, 1].mean()
        b = region[:, :, 2].mean()
        r_std = region[:, :, 0].std()
        g_std = region[:, :, 1].std()

        # White/light → rice or curd
        if r > 190 and g > 190 and b > 185:
            if b > 200 and abs(r - g) < 15:
                found.add("curd")
            else:
                found.add("rice")

        # Bright yellow/orange → dal
        if r > 175 and g > 115 and b < 85 and r > g * 1.2:
            found.add("dal")

        # Deep orange-red → rajma, tomato curry
        if r > 155 and g < 105 and b < 85:
            found.add("rajma")

        # Golden/tan + texture → roti
        if 135 < r < 215 and 105 < g < 180 and 55 < b < 135 and r > g > b and r_std > 15:
            found.add("roti")

        # Dark/olive green → sabzi
        if g > r and g > b and 45 < g < 160 and r < 135:
            found.add("sabzi")

        # Bright green → chutney/peas
        if g > 135 and g > r * 1.35 and g > b * 1.35:
            found.add("chutney")

        # Brown with high texture → chicken/meat curry
        if 95 < r < 190 and 65 < g < 150 and 35 < b < 120 and r > g > b and r_std > 28:
            found.add("chicken curry")

        # Yellow-grainy → khichdi/poha
        if 155 < r < 235 and 145 < g < 220 and b < 135 and abs(r - g) < 45 and r_std > 10:
            found.add("khichdi")

        # Creamy yellow → paneer
        if r > 195 and 165 < g < 230 and 95 < b < 185 and r > b and abs(r - g) < 45:
            found.add("paneer")

        # Reddish liquid → rasam/tomato soup
        if r > 145 and g < 95 and 55 < b < 125 and r_std < 22:
            found.add("rasam")

        # Dark gravy brown → curry/gravy
        if 80 < r < 160 and 50 < g < 120 and 30 < b < 100 and r > g and r_std > 20:
            found.add("curry")

    result_list = list(found)

    # Thali inference: common combinations
    if "rice" in result_list and "dal" not in result_list and len(result_list) >= 2:
        result_list.append("dal")
    if "roti" in result_list and "sabzi" not in result_list and len(result_list) >= 2:
        result_list.append("sabzi")
    if "rice" in result_list and "roti" in result_list and "curry" not in result_list:
        result_list.append("curry")

    # Remove "curry" if more specific items already cover it
    if "curry" in result_list and ("rajma" in result_list or "dal" in result_list or "chicken curry" in result_list):
        result_list.remove("curry")

    return result_list[:6] if result_list else ["mixed meal"]


def _mobilenet_classify(image: Image.Image, top_k=5) -> list:
    img_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        out = classifier(img_tensor)
    probs = torch.softmax(out, dim=1)[0]
    top_probs, top_idx = torch.topk(probs, top_k)
    found = []
    for prob, idx in zip(top_probs.tolist(), top_idx.tolist()):
        if prob < 0.05:
            continue
        if idx in IMAGENET_FOOD_MAP:
            lbl = IMAGENET_FOOD_MAP[idx]
            if lbl not in found:
                found.append(lbl)
    return found


def detect_food_items(image: Image.Image, api_key: str = ""):
    img_np = np.array(image.convert("RGB"))
    results = yolo_model(img_np, verbose=False)[0]

    yolo_confirmed = set()  # items YOLO found with high confidence
    detected = []
    boxes_info = []
    yolo_found_something = False

    if results.boxes is not None:
        for box in results.boxes:
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            if conf < 0.25:
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            if cls_id in FOOD_COCO:
                lbl = FOOD_COCO[cls_id]
                detected.append(lbl)
                boxes_info.append((lbl, conf, x1, y1, x2, y2))
                yolo_confirmed.add(lbl)
                yolo_found_something = True

            elif cls_id in CONTAINER_IDS and x2 > x1 and y2 > y1:
                crop = image.crop((x1, y1, x2, y2))
                color_labels = _analyze_region(crop)
                for lbl in color_labels[:3]:
                    if lbl != "mixed meal":
                        detected.append(lbl)
                        boxes_info.append((lbl, conf, x1, y1, x2, y2))
                        yolo_found_something = True

    # Color analysis on full image — primary Indian food detector
    color_items = _analyze_region(image)

    # MobileNet on full image — but filter Western false positives
    mn_items_raw = _mobilenet_classify(image, top_k=7)
    mn_items = []
    for item in mn_items_raw:
        # Only keep Western items if YOLO also confirmed them
        if item in WESTERN_FALSE_POSITIVES:
            if item in yolo_confirmed:
                mn_items.append(item)
            # else: skip — likely a false positive on Indian food
        else:
            mn_items.append(item)

    if not yolo_found_something:
        # Pure color + filtered mobilenet
        all_items = color_items + [i for i in mn_items if i not in color_items]
        w, h = image.size
        boxes_info = [(lbl, 0.8, 10, 10 + i*44, 240, 46 + i*44) for i, lbl in enumerate(all_items[:8])]
    else:
        all_items = detected + \
                    [i for i in color_items if i not in detected] + \
                    [i for i in mn_items if i not in detected and i not in color_items]

    # Deduplicate, remove "fried items" → replace with "roti" or "sabzi" if context allows
    seen, unique, unique_boxes = set(), [], []
    for item, box in zip(all_items, boxes_info):
        # Rename generic labels to more specific Indian ones
        if item == "fried items":
            if "roti" in all_items or "sabzi" in all_items:
                continue  # already covered by a better label
            else:
                item = "roti"  # most likely in Indian context

        if item not in seen and item != "mixed meal":
            seen.add(item)
            unique.append(item)
            unique_boxes.append(box)

    if not unique:
        unique = ["mixed meal"]
        unique_boxes = [("mixed meal", 0.8, 10, 10, 220, 50)]

    annotated = _draw_boxes(image, unique_boxes[:8])
    return unique[:8], annotated


def _draw_boxes(image, boxes_info):
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)
    w, h = image.size
    for idx, (label, conf, x1, y1, x2, y2) in enumerate(boxes_info):
        color = COLORS[idx % len(COLORS)]
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        tw = len(label) * 8 + 14
        draw.rectangle([x1, max(0, y1 - 24), min(w, x1 + tw), y1], fill=color)
        draw.text((x1 + 4, max(0, y1 - 21)), f"{label} {int(conf*100)}%", fill="white")
    return annotated
