from ultralytics.models.sam import SAM3VideoSemanticPredictor

# Initialize semantic video predictor
overrides = dict(conf=0.60, task="segment", mode="predict", imgsz=640, model="sam3.pt", half=True, save=True)
predictor = SAM3VideoSemanticPredictor(overrides=overrides)

# Track concepts using text prompts
# results = predictor(source="./video_raw/Raie_event_2.0.mp4", text=["a stingray", "a squid"], stream=True)

# # Process results
# for r in results:
#     r.show()  # Display frame with tracked objects

# Alternative: Track with bounding box prompts
results = predictor(
    source="./video_raw/Raie_event_2.0.mp4",
    bboxes=[[53, 372, 272, 485], [1151, 163, 1222, 199]],
    labels=[1, 1],  # Positive labels
    stream=False,
)