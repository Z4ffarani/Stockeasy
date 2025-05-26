import os
import json
import cv2
import glob
from ultralytics import YOLO
import yaml

ARQ_ESTOQUE   = "estoque.json"
MODEL_PATH    = "model.pt"           
GEN_MODEL     = "yolov8n.pt"          
DATA_YAML     = "dataset/data.yaml"
DATA_DIR      = "dataset"
IMG_TRAIN_DIR = os.path.join(DATA_DIR, "images/train")
LBL_TRAIN_DIR = os.path.join(DATA_DIR, "labels/train")
EPOCHS        = 3

os.makedirs(IMG_TRAIN_DIR, exist_ok=True)
os.makedirs(LBL_TRAIN_DIR, exist_ok=True)

def carregar_estoque():
    return json.load(open(ARQ_ESTOQUE)) if os.path.exists(ARQ_ESTOQUE) else {}

def salvar_estoque(d):
    json.dump(d, open(ARQ_ESTOQUE, "w"), indent=4)

if not os.path.exists(DATA_YAML):
    data_cfg = {
        "path": DATA_DIR,
        "train": "images/train",
        "val": "images/train",
        "nc": 0,
        "names": []
    }
    with open(DATA_YAML, "w") as f:
        yaml.dump(data_cfg, f)
else:
    with open(DATA_YAML) as f:
        data_cfg = yaml.safe_load(f)

gen_model = YOLO(GEN_MODEL)
model     = YOLO(MODEL_PATH)
estoque   = carregar_estoque()

def capturar_novas_amostras(classe, n=100):
    existing = len(glob.glob(os.path.join(IMG_TRAIN_DIR, f"{classe}_*.jpg")))
    cap = cv2.VideoCapture(0)
    saved = 0
    while saved < n:
        ret, frame = cap.read()
        if not ret:
            break
        res = gen_model(frame)[0]
        if len(res.boxes):
            box = max(res.boxes, key=lambda b: b.conf[0])
            x_c, y_c, w, h = box.xywh[0]
            H, W, _ = frame.shape
            cx, cy = x_c / W, y_c / H
            wn, hn = w / W, h / H
            idx = data_cfg["names"].index(classe)
            img_path = os.path.join(IMG_TRAIN_DIR, f"{classe}_{existing+saved}.jpg")
            lbl_path = os.path.join(LBL_TRAIN_DIR, f"{classe}_{existing+saved}.txt")
            cv2.imwrite(img_path, frame)
            with open(lbl_path, "w") as f:
                f.write(f"{idx} {cx:.6f} {cy:.6f} {wn:.6f} {hn:.6f}")
            saved += 1
            print(f"[{saved}/{n}] {classe} capturado.")
        cv2.imshow("Captura (q para sair)", frame)
        if cv2.waitKey(1) == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

def atualizar_data_yaml(classe):
    if classe not in data_cfg["names"]:
        data_cfg["names"].append(classe)
        data_cfg["nc"] = len(data_cfg["names"])
        with open(DATA_YAML, "w") as f:
            yaml.dump(data_cfg, f)
        print(f"Classe '{classe}' adicionada ao dataset.")

def fine_tune_model():
    print("🔁 Iniciando fine-tuning...")
    model.train(data=DATA_YAML, epochs=EPOCHS, imgsz=640, pretrained=True)
    os.replace("runs/detect/train/weights/best.pt", MODEL_PATH)
    print(f"✅ Fine-tuning concluído. Modelo salvo em {MODEL_PATH}.")

def detectar_e_contar(classe_alvo):
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)
    encontrado = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        results = model(frame, verbose=False, stream=False)
        for result in results:
            for box in result.boxes:
                class_id = int(box.cls[0])
                class_name = result.names[class_id]
                conf = float(box.conf[0])
                if class_name.lower() == classe_alvo.lower():
                    print(f"\n[✓] Detectado: {class_name} | Confiança: {conf:.2f}")
                    encontrado = True
                    break
        if encontrado or cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    return encontrado

def main():
    while True:
        print("\n📦 Stockeasy")
        print("[1] Cadastro de medicamento")
        print("[2] Entrada de insumo")
        print("[3] Remessa de insumo")
        print("[4] Visualizar estoque")
        print("[0] Sair\n")

        opcao = input("Escolha uma opção: ").strip()

        if opcao == "1":
            print("\n🔹 Cadastro de Medicamento")
            classe = input("Digite o nome do medicamento (ou 0 para voltar): ").strip().lower()
            if classe == "0":
                continue
            if classe:
                if classe not in estoque:
                    print("\n🆕 Novo medicamento detectado.")
                    estoque[classe] = 0
                    atualizar_data_yaml(classe)
                    capturar_novas_amostras(classe)
                    fine_tune_model()
                    print(f"\n✅ Medicamento '{classe}' cadastrado com sucesso!")
                else:
                    print(f"\n⚠️ Medicamento '{classe}' já está cadastrado.")
            else:
                print("\n❌ Nome inválido. Tente novamente.")

        elif opcao in ["2", "3"]:
            print("\n📷 Aguardando detecção do medicamento...")

            cap = cv2.VideoCapture(0)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)

            reconhecido = None

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                results = model(frame, verbose=False, stream=False)
                for result in results:
                    if result.boxes:
                        box = max(result.boxes, key=lambda b: b.conf[0])
                        class_id = int(box.cls[0])
                        conf = float(box.conf[0])
                        classe_detectada = result.names[class_id]

                        if classe_detectada in estoque:
                            reconhecido = classe_detectada
                            print(f"\n[✓] Detectado: {classe_detectada} | Confiança: {conf:.2f}")
                            break

                if reconhecido:
                    break

            cap.release()
            cv2.destroyAllWindows()

            if reconhecido:
                if opcao == "2":
                    estoque[reconhecido] += 1
                    print(f"\n[+] Entrada registrada")
                    print(f"📦 Estoque de '{reconhecido}': {estoque[reconhecido]}")
                elif opcao == "3":
                    if estoque[reconhecido] > 0:
                        estoque[reconhecido] -= 1
                        print(f"\n[-] Remessa registrada")
                        print(f"📦 Estoque de '{reconhecido}': {estoque[reconhecido]}")
                    else:
                        print(f"\n⚠️ Estoque de '{reconhecido}' insuficiente.")
                salvar_estoque(estoque)
            elif reconhecido is None:
                print("\n❌ Nenhum medicamento reconhecido.")

        elif opcao == "4":
            print("\n📊 Estoque Atual")
            if estoque:
                for med, qtd in estoque.items():
                    print(f"• {med}: {qtd} unidade(s)")
            else:
                print("⚠️ Nenhum medicamento cadastrado.")

        elif opcao == "0":
            print("\n👋 Encerrando o programa. Até logo!")
            break

        else:
            print("\n❌ Opção inválida. Tente novamente.")

if __name__ == "__main__":
    main()