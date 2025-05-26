import cv2
from ultralytics import YOLO

MODEL_PATH = "model.pt"
model = YOLO(MODEL_PATH)

def detectar_e_mostrar_classes():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, verbose=False)

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                class_id = int(box.cls[0])
                class_name = result.names[class_id]
                conf = float(box.conf[0])

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"{class_name} {conf:.2f}"
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                print(f"[✓] Detectado: {class_name} | Confiança: {conf:.2f}")

        cv2.imshow("Stockeasy", frame)

        if cv2.waitKey(1) == 27:  # Tecla Esc
            break

    cap.release()
    cv2.destroyAllWindows()

def main():
    while True:
        print("\n📦 Stockeasy | Classificação")
        print("[1] Iniciar detecção")
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