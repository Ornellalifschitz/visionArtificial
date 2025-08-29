import cv2
import time
import mediapipe as mp
import numpy as np
import os
import json

# Configuración de Mediapipe (basado en tutorial.py y ok_test.py)
mp_holistic = mp.solutions.holistic
holistic_model = mp_holistic.Holistic(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
 
mp_drawing = mp.solutions.drawing_utils

# Colores para la interfaz
RED_COLOR = (0, 0, 255)
GREEN_COLOR = (0, 255, 0)
BLUE_COLOR = (255, 0, 0)
YELLOW_COLOR = (0, 255, 255)

# Variables globales para el video
video_path = "/Users/ornellalifschitz/Downloads/IMG_0484.MOV"
video_capture = None
is_video_playing = False
is_video_paused = False

# Archivo para guardar la ruta del video
CONFIG_FILE = "video_config.json"

def load_video_path():
    """Carga la ruta del video desde el archivo de configuración"""
    global video_path
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, 'r') as f:
            config = json.load(f)
            video_path = config.get('video_path')
            return video_path
    return None

def save_video_path(path):
    """Guarda la ruta del video en el archivo de configuración"""
    with open(CONFIG_FILE, 'w') as f:
        json.dump({'video_path': path}, f)

def get_video_path():
    """Solicita la ruta del video al usuario si no está guardada"""
    global video_path
    if video_path is None:
        video_path = load_video_path()
    
    if video_path is None or not os.path.exists(video_path):
        video_path = input("Ingrese la ruta completa del archivo de video: ")
        if os.path.exists(video_path):
            save_video_path(video_path)
        else:
            print("Archivo no encontrado. Por favor, verifique la ruta.")
            return None
    
    return video_path

def calculate_distance(p1, p2):
    """Calcula la distancia entre dos puntos (basado en ok_test.py)"""
    return np.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

def is_open_palm(hand_landmarks):
    """Detecta si la mano está abierta (palma abierta)"""
    if not hand_landmarks:
        return False
    
    # Obtener landmarks de las puntas de los dedos
    thumb_tip = hand_landmarks.landmark[mp_holistic.HandLandmark.THUMB_TIP]
    index_tip = hand_landmarks.landmark[mp_holistic.HandLandmark.INDEX_FINGER_TIP]
    middle_tip = hand_landmarks.landmark[mp_holistic.HandLandmark.MIDDLE_FINGER_TIP]
    ring_tip = hand_landmarks.landmark[mp_holistic.HandLandmark.RING_FINGER_TIP]
    pinky_tip = hand_landmarks.landmark[mp_holistic.HandLandmark.PINKY_TIP]
    
    # Obtener landmarks de las articulaciones medias
    thumb_ip = hand_landmarks.landmark[mp_holistic.HandLandmark.THUMB_IP]
    index_pip = hand_landmarks.landmark[mp_holistic.HandLandmark.INDEX_FINGER_PIP]
    middle_pip = hand_landmarks.landmark[mp_holistic.HandLandmark.MIDDLE_FINGER_PIP]
    ring_pip = hand_landmarks.landmark[mp_holistic.HandLandmark.RING_FINGER_PIP]
    pinky_pip = hand_landmarks.landmark[mp_holistic.HandLandmark.PINKY_PIP]
    
    # Obtener la base de la palma
    wrist = hand_landmarks.landmark[mp_holistic.HandLandmark.WRIST]
    
    # Verificar si todos los dedos están extendidos
    # Un dedo está extendido si la punta está más lejos de la muñeca que la articulación media
    thumb_extended = thumb_tip.y < thumb_ip.y
    index_extended = index_tip.y < index_pip.y
    middle_extended = middle_tip.y < middle_pip.y
    ring_extended = ring_tip.y < ring_pip.y
    pinky_extended = pinky_tip.y < pinky_pip.y
    
    return all([thumb_extended, index_extended, middle_extended, ring_extended, pinky_extended])



def is_v_sign(hand_landmarks):
    """Detecta el gesto de V (dedos índice y medio extendidos)"""
    if not hand_landmarks:
        return False
    
    # Obtener landmarks de las puntas de los dedos
    index_tip = hand_landmarks.landmark[mp_holistic.HandLandmark.INDEX_FINGER_TIP]
    middle_tip = hand_landmarks.landmark[mp_holistic.HandLandmark.MIDDLE_FINGER_TIP]
    ring_tip = hand_landmarks.landmark[mp_holistic.HandLandmark.RING_FINGER_TIP]
    pinky_tip = hand_landmarks.landmark[mp_holistic.HandLandmark.PINKY_TIP]
    
    # Obtener landmarks de las articulaciones medias
    index_pip = hand_landmarks.landmark[mp_holistic.HandLandmark.INDEX_FINGER_PIP]
    middle_pip = hand_landmarks.landmark[mp_holistic.HandLandmark.MIDDLE_FINGER_PIP]
    ring_pip = hand_landmarks.landmark[mp_holistic.HandLandmark.RING_FINGER_PIP]
    pinky_pip = hand_landmarks.landmark[mp_holistic.HandLandmark.PINKY_PIP]
    
    # Verificar que índice y medio estén extendidos
    index_extended = index_tip.y < index_pip.y
    middle_extended = middle_tip.y < middle_pip.y
    
    # Verificar que anular y meñique estén cerrados
    ring_closed = ring_tip.y > ring_pip.y
    pinky_closed = pinky_tip.y > pinky_pip.y
    
    return index_extended and middle_extended and ring_closed and pinky_closed



def is_thumbs_up(hand_landmarks):
    """Detecta el gesto de pulgar hacia arriba"""
    if not hand_landmarks:
        return False
    
    # Obtener landmarks del pulgar
    thumb_tip = hand_landmarks.landmark[mp_holistic.HandLandmark.THUMB_TIP]
    thumb_ip = hand_landmarks.landmark[mp_holistic.HandLandmark.THUMB_IP]
    thumb_mcp = hand_landmarks.landmark[mp_holistic.HandLandmark.THUMB_MCP]
    
    # Obtener landmarks de otros dedos
    index_tip = hand_landmarks.landmark[mp_holistic.HandLandmark.INDEX_FINGER_TIP]
    middle_tip = hand_landmarks.landmark[mp_holistic.HandLandmark.MIDDLE_FINGER_TIP]
    ring_tip = hand_landmarks.landmark[mp_holistic.HandLandmark.RING_FINGER_TIP]
    pinky_tip = hand_landmarks.landmark[mp_holistic.HandLandmark.PINKY_TIP]
    
    index_pip = hand_landmarks.landmark[mp_holistic.HandLandmark.INDEX_FINGER_PIP]
    middle_pip = hand_landmarks.landmark[mp_holistic.HandLandmark.MIDDLE_FINGER_PIP]
    ring_pip = hand_landmarks.landmark[mp_holistic.HandLandmark.RING_FINGER_PIP]
    pinky_pip = hand_landmarks.landmark[mp_holistic.HandLandmark.PINKY_PIP]
    
    # Verificar que el pulgar esté extendido hacia arriba
    thumb_up = thumb_tip.y < thumb_ip.y < thumb_mcp.y
    
    # Verificar que otros dedos estén cerrados
    index_closed = index_tip.y > index_pip.y
    middle_closed = middle_tip.y > middle_pip.y
    ring_closed = ring_tip.y > ring_pip.y
    pinky_closed = pinky_tip.y > pinky_pip.y
    
    return thumb_up and all([index_closed, middle_closed, ring_closed, pinky_closed])

def open_video():
    """Abre el video"""
    global video_capture, is_video_playing
    if video_capture is None:
        path = get_video_path()
        if path:
            video_capture = cv2.VideoCapture(path)
            if video_capture.isOpened():
                is_video_playing = True
                print("Video abierto correctamente")
            else:
                print("Error al abrir el video")
                video_capture = None



def toggle_pause():
    """Alterna entre pausar y reproducir el video"""
    global is_video_paused
    if is_video_playing:
        is_video_paused = not is_video_paused
        status = "pausado" if is_video_paused else "reproduciendo"
        print(f"Video {status}")

def main():
    """Función principal del programa"""
    global video_capture, is_video_playing, is_video_paused
    
    # Inicializar la cámara (basado en tutorial.py)
    capture = cv2.VideoCapture(0)
    
    # Variables para debounce
    last_gesture_time = 0
    gesture_cooldown = 1.0  # 1 segundo entre gestos
    
    print("Iniciando control de video por gestos...")
    print("Gestos disponibles:")
    print("- Palma abierta: Abrir video")
    print("- Signo V: Pausar video (mantiene pausado)")
    print("- Pulgar arriba: Despausar video (mantiene reproduciendo)")
    print("Presiona 'q' para salir")
    
    while capture.isOpened():
        ret, frame = capture.read()
        
        if not ret:
            break
        
        # Convertir de BGR a RGB (basado en tutorial.py)
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Procesar con Mediapipe
        image.flags.writeable = False
        results = holistic_model.process(image)
        image.flags.writeable = True
        
        # Convertir de vuelta a BGR
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Detectar gestos en ambas manos
        right_open_palm = False
        left_open_palm = False
        right_v_sign = False
        left_v_sign = False
        right_thumbs_up = False
        left_thumbs_up = False
        
        current_time = time.time()
        
        # Procesar mano derecha
        if results.right_hand_landmarks:
            right_open_palm = is_open_palm(results.right_hand_landmarks)
            right_v_sign = is_v_sign(results.right_hand_landmarks)
            right_thumbs_up = is_thumbs_up(results.right_hand_landmarks)
            
            # Dibujar landmarks de la mano derecha
            mp_drawing.draw_landmarks(
                image, 
                results.right_hand_landmarks, 
                mp_holistic.HAND_CONNECTIONS
            )
        
        # Procesar mano izquierda
        if results.left_hand_landmarks:
            left_open_palm = is_open_palm(results.left_hand_landmarks)
            left_v_sign = is_v_sign(results.left_hand_landmarks)
            left_thumbs_up = is_thumbs_up(results.left_hand_landmarks)
            
            # Dibujar landmarks de la mano izquierda
            mp_drawing.draw_landmarks(
                image, 
                results.left_hand_landmarks, 
                mp_holistic.HAND_CONNECTIONS
            )
        
        # Procesar gestos con debounce
        if current_time - last_gesture_time > gesture_cooldown:
            # Abrir video con palma abierta
            if (right_open_palm or left_open_palm) and not is_video_playing:
                open_video()
                last_gesture_time = current_time
            
            # Pausar con signo V (mantiene pausado indefinidamente)
            elif (right_v_sign or left_v_sign) and is_video_playing and not is_video_paused:
                toggle_pause()
                last_gesture_time = current_time
            
            # Despausar con pulgar arriba (mantiene reproduciendo indefinidamente)
            elif (right_thumbs_up or left_thumbs_up) and is_video_playing and is_video_paused:
                toggle_pause()
                last_gesture_time = current_time
        
        # Mostrar información en pantalla
        y_offset = 30
        cv2.putText(image, f"Video: {'Reproduciendo' if is_video_playing and not is_video_paused else 'Pausado' if is_video_paused else 'Cerrado'}", 
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, GREEN_COLOR, 2)
        
        # Mostrar gestos detectados
        gesture_text = []
        if right_open_palm or left_open_palm:
            gesture_text.append("Palma Abierta")
        if right_v_sign or left_v_sign:
            gesture_text.append("Signo V")
        if right_thumbs_up or left_thumbs_up:
            gesture_text.append("Pulgar Arriba")
        
        if gesture_text:
            cv2.putText(image, f"Gesto: {', '.join(gesture_text)}", 
                       (10, y_offset + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, BLUE_COLOR, 2)
        
        # Mostrar instrucciones
        cv2.putText(image, "Palma abierta: Abrir | V: Pausar | Pulgar arriba: Despausar", 
                   (10, image.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, YELLOW_COLOR, 1)
        
        # Mostrar la imagen de la cámara
        cv2.imshow("Control de Video por Gestos", image)
        
        # Reproducir video si está abierto y no está pausado
        if is_video_playing and video_capture is not None and not is_video_paused:
            video_ret, video_frame = video_capture.read()
            if video_ret:
                # Rotar el video 90 grados hacia la izquierda (antihorario)
                rotated_frame = cv2.rotate(video_frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
                
                # Obtener dimensiones de la pantalla
                screen_width = 1920  # Ancho típico de pantalla
                screen_height = 1080  # Alto típico de pantalla
                
                # Calcular tamaño para la mitad de la pantalla
                target_width = screen_width // 2
                target_height = screen_height // 2
                
                # Redimensionar el video manteniendo proporción
                video_height, video_width = rotated_frame.shape[:2]
                aspect_ratio = video_width / video_height
                
                if aspect_ratio > 1:  # Video más ancho que alto
                    new_width = target_width
                    new_height = int(target_width / aspect_ratio)
                else:  # Video más alto que ancho
                    new_height = target_height
                    new_width = int(target_height * aspect_ratio)
                
                # Redimensionar el video
                resized_frame = cv2.resize(rotated_frame, (new_width, new_height))
                cv2.imshow("Video Reproduciéndose", resized_frame)
            else:
                # Si el video terminó, reiniciarlo
                video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        # Salir con 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Limpiar recursos
    capture.release()
    if video_capture is not None:
        video_capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()