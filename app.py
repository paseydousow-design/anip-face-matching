from functools import lru_cache
import os
from typing import Any, Dict, Tuple

# Helps local Windows/Anaconda environments that load two OpenMP runtimes.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import gradio as gr
import numpy as np
import torch
import torch.nn.functional as F
from facenet_pytorch import InceptionResnetV1, MTCNN
from PIL import Image


THRESHOLD_DEFAULT = 0.554
IMAGE_SIZE = 160
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@lru_cache(maxsize=1)
def load_models() -> Tuple[MTCNN, InceptionResnetV1]:
    mtcnn = MTCNN(
        image_size=IMAGE_SIZE,
        margin=20,
        post_process=True,
        device=DEVICE,
    )
    facenet = InceptionResnetV1(pretrained="vggface2", classify=False).eval().to(DEVICE)
    for param in facenet.parameters():
        param.requires_grad = False
    return mtcnn, facenet


def fallback_preprocess(image: Image.Image) -> torch.Tensor:
    image = image.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
    arr = np.asarray(image).astype("float32")
    arr = (arr - 127.5) / 128.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1)
    return tensor


def aligned_tensor(image: Image.Image) -> Tuple[torch.Tensor, bool]:
    mtcnn, _ = load_models()
    face = mtcnn(image.convert("RGB"))
    if face is None:
        return fallback_preprocess(image), False
    return face, True


def make_views(image: Image.Image, use_tta: bool) -> list[Image.Image]:
    image = image.convert("RGB")
    if not use_tta:
        return [image]
    return [
        image,
        image.transpose(Image.Transpose.FLIP_LEFT_RIGHT),
        image.rotate(7, resample=Image.Resampling.BILINEAR),
        image.rotate(-7, resample=Image.Resampling.BILINEAR),
    ]


@torch.no_grad()
def embed_image(image: Image.Image, use_tta: bool) -> Tuple[torch.Tensor, Dict[str, Any]]:
    _, facenet = load_models()
    tensors = []
    aligned_count = 0

    for view in make_views(image, use_tta):
        tensor, aligned = aligned_tensor(view)
        tensors.append(tensor)
        aligned_count += int(aligned)

    batch = torch.stack(tensors).to(DEVICE)
    embeddings = facenet(batch)
    embedding = embeddings.mean(dim=0, keepdim=True)
    embedding = F.normalize(embedding, p=2, dim=1)

    diagnostics = {
        "views": len(tensors),
        "aligned_views": aligned_count,
        "fallback_views": len(tensors) - aligned_count,
        "device": str(DEVICE),
        "embedding_dim": int(embedding.shape[1]),
    }
    return embedding.cpu(), diagnostics


def compare_faces(
    image_1: Image.Image | None,
    image_2: Image.Image | None,
    threshold: float,
    use_tta: bool,
) -> Tuple[str, float, Dict[str, Any]]:
    if image_1 is None or image_2 is None:
        return "Ajoute deux images pour lancer la comparaison.", 0.0, {}

    emb_1, diag_1 = embed_image(image_1, use_tta)
    emb_2, diag_2 = embed_image(image_2, use_tta)

    similarity = float(torch.sum(emb_1 * emb_2).item())
    is_match = similarity >= threshold

    verdict = "MATCH" if is_match else "PAS MATCH"
    color = "green" if is_match else "red"
    confidence_gap = abs(similarity - threshold)

    message = (
        f"## <span style='color:{color}'>{verdict}</span>\n"
        f"Similarite cosinus : **{similarity:.4f}**\n\n"
        f"Seuil utilise : **{threshold:.3f}**\n\n"
        f"Ecart au seuil : **{confidence_gap:.4f}**"
    )

    diagnostics = {
        "model": "FaceNet VGGFace2",
        "alignment": "MTCNN avec fallback resize",
        "tta_enabled": bool(use_tta),
        "image_1": diag_1,
        "image_2": diag_2,
    }
    return message, similarity, diagnostics


with gr.Blocks(theme=gr.themes.Soft(), title="ANIP Face Matching") as demo:
    gr.Markdown(
        """
        # ANIP - Tache 1 : Face Matching

        Demo de reconnaissance faciale issue du challenge ANIP Benin 2024.
        Upload deux images, le modele calcule des embeddings FaceNet puis compare
        les deux visages avec une similarite cosinus.
        """
    )

    with gr.Row():
        image_1 = gr.Image(type="pil", label="Image 1")
        image_2 = gr.Image(type="pil", label="Image 2")

    with gr.Row():
        threshold = gr.Slider(
            minimum=0.20,
            maximum=0.90,
            value=THRESHOLD_DEFAULT,
            step=0.001,
            label="Seuil de decision",
        )
        use_tta = gr.Checkbox(value=True, label="Activer TTA leger")

    run_btn = gr.Button("Comparer les deux visages", variant="primary")

    result = gr.Markdown(label="Resultat")
    similarity = gr.Number(label="Similarite cosinus", precision=4)
    diagnostics = gr.JSON(label="Diagnostics")

    gr.Markdown(
        """
        ### Methode
        - Alignement facial : MTCNN
        - Backbone : FaceNet VGGFace2
        - Embedding : 512 dimensions, normalise L2
        - Decision : similarite cosinus >= seuil

        Code source : https://github.com/paseydousow-design/anip-face-matching
        """
    )

    run_btn.click(
        fn=compare_faces,
        inputs=[image_1, image_2, threshold, use_tta],
        outputs=[result, similarity, diagnostics],
    )


if __name__ == "__main__":
    demo.launch()
