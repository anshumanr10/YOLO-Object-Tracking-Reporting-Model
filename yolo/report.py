def write_summary():
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    output_file = "detections_summary.txt"

    with open(output_file, "w") as f:
        f.write("YOLO Detection Summary\n")
        f.write("======================\n")
        f.write(f"Timestamp: {timestamp}\n\n")

        f.write("Frame-level detections (counts per frame):\n")
        if frame_detection_counts:
            for cls in sorted(frame_detection_counts.keys()):
                f.write(f"{cls}: {frame_detection_counts[cls]}\n")
        else:
            f.write("None\n")

        f.write("\nUnique tracked objects (by track ID):\n")
        if unique_ids_by_class:
            for cls in sorted(unique_ids_by_class.keys()):
                f.write(f"{cls}: {len(unique_ids_by_class[cls])}\n")
        else:
            f.write("None\n")

        f.write("\nDetected classes:\n")
        if detected_classes:
            for cls in sorted(detected_classes):
                f.write(f"- {cls}\n")
        else:
            f.write("- None\n")

    print(f"\nSummary written to {output_file}")