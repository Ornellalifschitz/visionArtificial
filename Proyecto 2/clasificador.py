import cv2
import numpy as np
import math
import joblib
from joblib import load

# Label conversion functions for musical notes
def int_to_label(string_label):
    if string_label == 1: 
        return 'corchea'
    elif string_label == 2: 
        return 'silencio'
    elif string_label == 3:
        return 'negra'
    else:
        return 'unknown'

# Contour detection functions
def get_contours(frame, mode, method):
    contours, hierarchy = cv2.findContours(frame, mode, method)
    return contours

def filter_contours_by_area(contours, min_area, max_area):
    filtered_contours = []
    for cnt in contours:
        if min_area <= cv2.contourArea(cnt) <= max_area:
            filtered_contours.append(cnt)
    return filtered_contours

def get_bounding_rect(contour):
    return cv2.boundingRect(contour)

# Frame processing functions
def apply_color_convertion(frame, color):
    return cv2.cvtColor(frame, color)

def threshold(frame, slider_max, binary, trackbar_value):
    _, th = cv2.threshold(frame, trackbar_value, slider_max, binary)
    return th

def denoise(frame, method, radius):
    kernel = cv2.getStructuringElement(method, (radius, radius))
    opening = cv2.morphologyEx(frame, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
    return closing

def draw_contours(frame, contours, color, thickness):
    cv2.drawContours(frame, contours, -1, color, thickness)
    return frame

# Trackbar functions
def create_trackbar(trackbar_name, window_name, slider_max):
    cv2.createTrackbar(trackbar_name, window_name, 0, slider_max, on_trackbar)

def on_trackbar(val):
    pass

def get_trackbar_value(trackbar_name, window_name):
    return int(cv2.getTrackbarPos(trackbar_name, window_name))

# Color definitions
COLOR_GREEN = (0, 255, 0)
COLOR_RED = (0, 0, 255)
COLOR_BLUE = (255, 0, 0)
COLOR_YELLOW = (0, 255, 255)

def main():
    window_name = 'Musical Notes Classifier'
    cv2.namedWindow(window_name)
    
    # Initialize camera
    cap = cv2.VideoCapture(0)
    
    # Load the pre-trained model
    try:
        clasificador = load('models/modelo_notas_musicales.joblib')
        print("✅ Modelo cargado exitosamente!")
    except FileNotFoundError:
        print("❌ Error: No se encontró el archivo del modelo 'models/modelo_notas_musicales.joblib'")
        print("Asegúrate de que el modelo esté entrenado y guardado correctamente.")
        return
    except Exception as e:
        print(f"❌ Error al cargar el modelo: {e}")
        return

    # Create trackbars for parameter adjustment
    trackbar_thresh_name = 'Threshold'
    thresh_slider_max = 255
    create_trackbar(trackbar_thresh_name, window_name, thresh_slider_max)

    trackbar_kernel_name = 'Kernel denoise'
    contour_kernel_max = 10
    create_trackbar(trackbar_kernel_name, window_name, contour_kernel_max)

    trackbar_min_area_name = 'Min Area'
    contour_min_area_max = 10000
    create_trackbar(trackbar_min_area_name, window_name, contour_min_area_max)

    trackbar_max_area_name = 'Max Area'
    contour_max_area_max = 99999
    create_trackbar(trackbar_max_area_name, window_name, contour_max_area_max)

    print("🎵 Clasificador de Notas Musicales iniciado!")
    print("📋 Controles:")
    print("  - Usa los trackbars para ajustar los parámetros")
    print("  - Presiona 'q' para salir")
    print("  - Muestra notas musicales frente a la cámara para clasificarlas")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Error: No se pudo capturar el frame de la cámara")
            break
        
        # Convert to grayscale
        gray_frame = apply_color_convertion(frame=frame, color=cv2.COLOR_BGR2GRAY)
        
        # Get trackbar values
        trackbar_thresh_val = get_trackbar_value(trackbar_name=trackbar_thresh_name, window_name=window_name)
        trackbar_min_area_val = get_trackbar_value(trackbar_name=trackbar_min_area_name, window_name=window_name)
        trackbar_max_area_val = get_trackbar_value(trackbar_name=trackbar_max_area_name, window_name=window_name)

        # Apply threshold
        thresh_frame = threshold(frame=gray_frame, slider_max=thresh_slider_max,
                                 binary=cv2.THRESH_BINARY,
                                 trackbar_value=trackbar_thresh_val)

        # Denoise the frame
        frame_denoised = denoise(frame=thresh_frame, method=cv2.MORPH_ELLIPSE, radius=5)

        # Find contours
        contours = get_contours(frame=frame_denoised, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)

        # Filter contours by area
        filtered_contours = filter_contours_by_area(contours=contours, min_area=trackbar_min_area_val,
                                                    max_area=trackbar_max_area_val)

        # Process each contour
        for cont in filtered_contours:
            try:
                # Calculate Hu moments
                mom = cv2.moments(cont)
                hu_moments = cv2.HuMoments(mom)
                
                # Apply logarithmic transformation to Hu moments
                for i in range(0, 7):
                    if (hu_moments[i] != 0):
                        hu_moments[i] = -1 * math.copysign(1.0, hu_moments[i]) * math.log10(abs(hu_moments[i]))
                
                # Prepare sample for prediction
                sample = np.array(hu_moments, dtype=np.float32).reshape(1, -1)
                
                # Make prediction
                predict = clasificador.predict(sample)[0]
                
                # Convert prediction to label
                label = int_to_label(predict)

                # Choose color based on prediction
                if label == 'corchea':
                    color = COLOR_BLUE
                elif label == 'silencio':
                    color = COLOR_RED
                elif label == 'negra':
                    color = COLOR_GREEN
                else:
                    color = COLOR_YELLOW

                # Draw contour and label
                draw_contours(frame=frame, contours=[cont], color=color, thickness=3)
                
                # Get bounding rectangle for text placement
                x, y, _, __ = get_bounding_rect(cont)
                cv2.putText(frame, label, (x - 20, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                
            except Exception as e:
                print(f"⚠️ Error procesando contorno: {e}")
                continue

        # Display frames
        cv2.imshow(window_name, frame)
        cv2.imshow('Debug - Threshold', frame_denoised)

        # Check for exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("👋 Clasificador cerrado correctamente")

if __name__ == "__main__":
    main()