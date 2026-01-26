# Draw
                label = f"{class_name} {conf_score:.2f}"
                if b.id is not None:
                    label += f" ID:{int(b.id.item())}"

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, max(20, y1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow("YOLO Webcam", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
cap.release()
    cv2.destroyAllWindows()
    write_summary()