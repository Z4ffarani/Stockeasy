import cv2
from ultralytics import YOLO

MODEL_PATH = "model.pt"

# Carregar o modelo treinado
model = YOLO(MODEL_PATH)

def detectar_e_mostrar_classes():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Realiza a detecção usando o modelo
        results = model(frame, verbose=False, stream=False)

        # Itera pelas caixas de detecção
        for result in results:
            for box in result.boxes:
                class_id = int(box.cls[0])
                class_name = result.names[class_id]
                conf = float(box.conf[0])

                # Exibe a classe detectada e a confiança no console
                print(f"\n[✓] Detectado: {class_name} | Confiança: {conf:.2f}")

        # Exibe o vídeo da webcam
        cv2.imshow("Detecção de Classes (Esc para sair)", frame)

        # Pressione 'Esc' para sair
        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

def main():
    while True:
        print("\n📦 Stockeasy - Reconhecimento de Classes")
        print("[1] Iniciar Detecção")
        print("[0] Sair\n")

        opcao = input("Escolha uma opção: ").strip()

        if opcao == "1":
            print("\n📷 Iniciando a detecção...")
            detectar_e_mostrar_classes()

        elif opcao == "0":
            print("\n👋 Encerrando o programa. Até logo!")
            break

        else:
            print("\n❌ Opção inválida. Tente novamente.")

if __name__ == "__main__":
    main()
