import cv2
import math
import os
from deepface import DeepFace

# отслеживание для меньших вычислений
class SmartFaceTracker:
    def __init__(self, db_path, recognition_interval=30, emotion_interval=5):
        self.db_path = db_path
        self.recognition_interval = recognition_interval
        self.emotion_interval = emotion_interval

        self.tracked_faces = []
        self.frame_count = 0

    def get_center(self, x, y, w, h):
        return (x + w // 2, y + h // 2)

    def get_distance(self, p1, p2):
        return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

    def process_frame(self, frame):
        self.frame_count += 1

        # Детекция лиц
        try:
            face_objs = DeepFace.extract_faces(
                img_path=frame,
                detector_backend='mtcnn',
                enforce_detection=False,
                align=False
            )
        except Exception as e:
            face_objs = []

        current_frame_faces = []

        for f_obj in face_objs:
            facial_area = f_obj['facial_area']
            x, y, w, h = facial_area['x'], facial_area['y'], facial_area['w'], facial_area['h']

            # Отфильтруем мусор
            if w < 30 or h < 30: continue

            center_new = self.get_center(x, y, w, h)

            matched_face = None

            # чтобы постоянно к базе не обращаться
            for cached in self.tracked_faces:
                center_old = self.get_center(*cached['coords'])
                dist = self.get_distance(center_new, center_old)

                if dist < 50:
                    matched_face = cached
                    break

            # ЛОГИКА ОБНОВЛЕНИЯ ДАННЫХ
            name = "Unknown"
            emotion = "neutral"

            should_recognize = (matched_face is None) or \
                               ((self.frame_count - matched_face['last_rec_frame']) > self.recognition_interval)

            should_update_emotion = (matched_face is None) or \
                                    ((self.frame_count - matched_face['last_emo_frame']) > self.emotion_interval)

            if matched_face:
                # Если лицо старое, берем данные из памяти по умолчанию
                name = matched_face['name']
                emotion = matched_face['emotion']
                rec_frame = matched_face['last_rec_frame']
                emo_frame = matched_face['last_emo_frame']
            else:
                rec_frame = 0
                emo_frame = 0

            # --- ТЯЖЕЛЫЙ БЛОК: Распознавание личности ---
            if should_recognize:
                # Вырезаем лицо для поиска
                face_img = frame[y:y + h, x:x + w]
                if face_img.size > 0:
                    try:
                        dfs = DeepFace.find(
                            img_path=face_img,
                            db_path=self.db_path,
                            model_name="VGG-Face", # модель
                            enforce_detection=False,
                            silent=True,
                            threshold=0.65  # порог узнаваемости чем больше тем сильнее
                        )

                        found_match = False
                        if len(dfs) > 0:
                            # DeepFace возвращает список датафреймов
                            df = dfs[0]
                            if not df.empty:
                                full_path = df.iloc[0]['identity']
                                filename = os.path.basename(full_path)
                                name = os.path.splitext(filename)[0]
                                found_match = True

                        if not found_match:
                            name = "Unknown"

                    except Exception as e:
                        print(f"Ошибка поиска в БД: {e}")
                        name = "Unknown"

                    rec_frame = self.frame_count

            # --- СРЕДНИЙ БЛОК: Анализ эмоций ---
            if should_update_emotion:
                face_img = frame[y:y + h, x:x + w]
                if face_img.size > 0:
                    try:
                        # analyze возвращает массив, берем первый элемент
                        emo_res = DeepFace.analyze(img_path=face_img, actions=['emotion'], enforce_detection=False,
                                                   silent=True)
                        if isinstance(emo_res, list):
                            emotion = emo_res[0]['dominant_emotion']
                        else:
                            emotion = emo_res['dominant_emotion']
                    except:
                        emotion = "neutral"
                    emo_frame = self.frame_count

            # Сохраняем обновленные данные в список текущего кадра
            current_frame_faces.append({
                'name': name,
                'coords': (x, y, w, h),
                'emotion': emotion,
                'last_rec_frame': rec_frame,
                'last_emo_frame': emo_frame
            })

            # ВИЗУАЛИЗАЦИЯ
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)

            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

            # Формируем текст
            label = f"{name}"
            if should_recognize: label += " *"

            cv2.putText(frame, label, (x, y - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.putText(frame, emotion, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Обновляем глобальный трекер
        self.tracked_faces = current_frame_faces
        return frame


# --- ЗАПУСК ---
if __name__ == "__main__":
    DB_FOLDER = "my_db"

    tracker = SmartFaceTracker(db_path=DB_FOLDER)
    cap = cv2.VideoCapture(0)

    print("Нажмите 'q' для выхода.")

    while True:
        ret, frame = cap.read()
        if not ret: break

        processed_frame = tracker.process_frame(frame)

        cv2.imshow("face & emotion", processed_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()